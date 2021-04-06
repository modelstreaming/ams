import os
import time
import numpy as np
import cv2
from collections import deque
import subprocess as sp
from ams.utils.utils import calculate_miou, string_class_iou, choose_frames
from ams.exp_configs import class_weights, test_length, coco_class_converter, is_coco
from ams.SemanticNetwork import SemanticNetwork

from termcolor import colored

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_video', None, 'Directory for the video')
flags.mark_flag_as_required('input_video')
flags.DEFINE_string('gt_video', None, 'Directory for the ground truth labels of video')
flags.mark_flag_as_required('gt_video')
flags.DEFINE_string('student_checkpoint', None, 'Directory for student checkpoint')
flags.mark_flag_as_required('student_checkpoint')
flags.DEFINE_string('output_dir', None, 'Directory for the output figure')
flags.mark_flag_as_required('output_dir')
flags.DEFINE_string('gpu', None, 'GPU to use for this')
flags.mark_flag_as_required('gpu')

flags.DEFINE_bool('initial_fill', False, 'When true, doesn\'t train until memory is full')
flags.DEFINE_integer('memory_len', 250, 'Memory length')
flags.DEFINE_integer('batch_size', 10, 'Mini batch size')
flags.DEFINE_integer('iter', 200, '# of iterations')
flags.DEFINE_integer('height', 256, 'height of video')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')

flags.DEFINE_integer('send_period', 30, 'Period between frame sample arrival')
flags.DEFINE_integer('train_period', 10, 'Training rate')

flags.DEFINE_bool('only_results', False, 'Just print the results')
flags.DEFINE_bool('compress_uplink', False, 'Compress the uplink using H264 encoding')
flags.DEFINE_bool('no_restore', False, 'Do not restore the model on every training')
flags.DEFINE_bool('save_pic', False, 'Save the pictures in inference')

flags.DEFINE_bool('enable_ASR', False, 'Enable Adaptive Sampling Rate')
flags.DEFINE_bool('enable_ATR', False, 'Enable Adaptive Training Rate')

flags.DEFINE_enum('train_strategy', 'full_model', ['full_model',
                                                   'coord_desc_auto',
                                                   'coord_desc_last',
                                                   'coord_desc_first',
                                                   'coord_desc_both',
                                                   'coord_desc_rand'],
                  'Strategy of selecting which parts of the model to retrain every time')
flags.DEFINE_enum('coord_fraction', '0.1', ['0.1',
                                            '0.05',
                                            '0.2',
                                            '0.01'],
                  'Fraction of parameters trained in coordinate descent mode')

flags.DEFINE_enum('mode', None, ['simple',
                                 'pretrained',
                                 'horizon',
                                 'early'],
                  'Profiling mode')
flags.mark_flag_as_required('mode')

flags.DEFINE_integer('early_cutoff_time', 60, 'Where to start making the one-time customized model')

SIZE = [FLAGS.height, FLAGS.height * 2]

assert not flags.enable_ATR or flags.enable_ASR, 'ASR must be enabled for ATR to work'
assert not flags.enable_ASR or flags.mode == 'simple', 'ASR can only be used in simple mode'
assert not flags.enable_ATR or flags.mode == 'simple', 'ATR can only be used in simple mode'


def train_model(train_start, train_end, sampling_period, gpu_id, run_label, gt_path, exp_num, save_range,
                sample_send_period):
    """
    This function emulates the training phase of the server-client setting. It collects frames with a rate of
    sampling_period in the range [train_start, train_end). It saves the trained models for time points in save_range

    :param train_start: Start of the interval
    :param train_end: End of the interval
    :param gpu_id: GPU index to use for the training
    :param sampling_period: Frame sampling rate
    :param run_label: A label used to recognize this experiment's output, must be unique to this experiment
    :param gt_path: Where ground truth labels are saved
    :param exp_num: A unique number assigned to each video that can be used to look up it's length and chosen classes
    :param save_range: The points to save the models
    :param sample_send_period: The period at which samples are sent, in seconds
    :type train_start: int
    :type train_end: int
    :type gpu_id: str
    :type sampling_period: int
    :type run_label: str
    :type gt_path: str
    :type exp_num: int
    :type save_range: list of int
    :type sample_send_period: int
    """
    assert train_end - train_start != 0, "There should be at least one set of data points"
    # Open video, get the fps and set it's starting point to train_start
    cap = cv2.VideoCapture(FLAGS.input_video)
    if not cap.isOpened():
        print_process("Error opening video stream or file", -1)
        exit(1)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    train_end_frame = train_end * fps
    i = train_start * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    # Initialize variables to track down-link bandwidth usage
    update_count = 0
    send_rate = sampling_period / fps
    sample_per_period = []
    up_bw_per_period = []  # in bits
    down_bw_per_period = []  # in bits
    frame_label_bucket = []
    num_unseen_frames = 0
    # ATR state variables and logs, if ATRis used, save_range changes in the middle of a run and must be saved
    model_save_times = [0]
    train_period_reset = save_range[2] - save_range[1]
    train_period_current = save_range[2] - save_range[1]
    if flags.enable_ATR:
        assert train_period_current == flags.train_period
        for j in range(2, len(save_range)):
            assert train_period_current == save_range[j] - save_range[j-1]
    send_rate_deq = deque(maxlen=5)
    hibernate = False
    # COCO labels need preprocessing, doesn't do anything if the dataset is not LVS
    map_coco = None
    if is_coco(exp_num):
        map_coco = coco_class_converter()
    # Use deques to keep a finite number of data points, representing a span of FLAGS.memory_len amount of seconds
    frame_memory = deque(maxlen=int(FLAGS.memory_len / sampling_period * fps))
    label_memory = deque(maxlen=int(FLAGS.memory_len / sampling_period * fps))
    to_compress_frame_memory = deque(maxlen=int(FLAGS.memory_len / sampling_period * fps))
    # Initialize the model
    semantic_network = SemanticNetwork(meta_dir=FLAGS.student_checkpoint,
                                       class_weights_exp=class_weights(exp_num),
                                       height=FLAGS.height,
                                       gpu_id=gpu_id,
                                       scale=[1],
                                       mini_batch_size=FLAGS.batch_size,
                                       lr=FLAGS.lr,
                                       mem_frac=1,
                                       coord_frac=float(FLAGS.coord_fraction),
                                       train_biases_only=False,
                                       regularize=False,
                                       masked_gradients=FLAGS.train_strategy not in ['full_model'],
                                       cross_miou_compat=flags.enable_ASR)
    # Initially save the model
    save_dir = get_save_dir(run_label + "_%d" % train_start)
    semantic_network.save_to_frozen_graph(save_dir + "_final")
    print_process("Saved model to %s_final.pb" % save_dir, 0)

    while cap.isOpened() and i < train_end_frame:
        # Read frame from video
        ret, frame = cap.read()
        if ret:
            # load corresponding label in gt_path
            gt = cv2.imread("%sgt_%06d.png" % (gt_path, i), cv2.IMREAD_GRAYSCALE)
            frame_label_bucket.append((frame, gt))
        else:
            print("Premature end of video, exiting")
            exit(1)

        i += 1
        if i % (5 * fps) == 0:
            print_process("%d seconds elapsed" % (i / fps), i / fps)

        if i // fps % sample_send_period == 0:
            # When it's time to send, choose frames to send based on send_rate
            frames_chosen, labels_chosen = choose_frames(frame_label_bucket, send_rate)
            for frame, label in zip(frames_chosen, labels_chosen):
                if FLAGS.compress_uplink:
                    # If we use compress_uplink, use twice the resolution to send a higher quality
                    frame = cv2.resize(frame, (SIZE[1] * 2, SIZE[0] * 2))
                else:
                    frame = cv2.resize(frame, (SIZE[1], SIZE[0]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                label_resized = cv2.resize(label, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_NEAREST)
                to_compress_frame_memory.append(frame)
                if map_coco is not None:
                    label_resized = map_coco[label_resized]
                label_memory.extend(label_resized)
            frame_label_bucket.clear()

            num_frames = len(to_compress_frame_memory)
            sample_per_period.append(num_frames)
            # Log unseen frames to use for computing phi-score in ASR
            num_unseen_frames += num_frames

            if FLAGS.compress_uplink:
                # First write frames to video files using FFMPEG, then read frames from that video and append them to
                # the server's memory
                output_video_file = f"{get_save_dir(run_label)}_tmp_movie.mp4"
                time_start_encode = time.time()
                trying = True
                while trying:
                    # If multiple runs are initiated, their input pipes can compete, so we add a while loop to keep
                    # trying until succeeded
                    try:
                        with open(os.devnull, "w") as f:
                            proc = sp.Popen(
                                ['/usr/bin/ffmpeg',
                                 '-y',
                                 '-s', '1024x512',
                                 '-pixel_format', 'bgr24',
                                 '-f', 'rawvideo',
                                 '-r', '10',
                                 '-i', 'pipe:',
                                 '-vcodec', 'libx264',
                                 '-pix_fmt', 'yuv420p',
                                 '-preset', 'medium',
                                 '-b:v', '%dk' % (FLAGS.uplink_bw * sample_send_period),
                                 '-pass', '1',
                                 '-f', 'mp4',
                                 '/dev/null'],
                                stdin=sp.PIPE, stderr=f, stdout=f)
                            for index_vi_frame in range(len(to_compress_frame_memory)):
                                frame_to_pipe = to_compress_frame_memory[index_vi_frame]
                                proc.stdin.write(frame_to_pipe.tobytes())
                            proc.stdin.close()
                            proc.wait()
                            proc.terminate()
                            proc = sp.Popen(
                                ['/usr/bin/ffmpeg',
                                 '-y',
                                 '-s', '1024x512',
                                 '-pixel_format', 'bgr24',
                                 '-f', 'rawvideo',
                                 '-r', '10',
                                 '-i', 'pipe:',
                                 '-vcodec', 'libx264',
                                 '-pix_fmt', 'yuv420p',
                                 '-preset', 'medium',
                                 '-b:v', '%dk' % (FLAGS.uplink_bw * sample_send_period),
                                 '-pass', '2',
                                 f'{output_video_file}'],
                                stdin=sp.PIPE, stderr=f, stdout=f)
                            for index_vi_frame in range(len(to_compress_frame_memory)):
                                frame_to_pipe = to_compress_frame_memory[index_vi_frame]
                                proc.stdin.write(frame_to_pipe.tobytes())
                            proc.stdin.close()
                            proc.wait()
                            proc.terminate()
                            trying = False
                    except BrokenPipeError:
                        print_process("GOT BROKEN PIPE, TRYING AGAIN", i // fps)
                        continue
                to_compress_frame_memory.clear()
                print("FFMPEG took %.1f ms to encode" % ((time.time() - time_start_encode) * 1000))
                size_vid = os.path.getsize(output_video_file) / 1024
                print_process("Video is %.2fKB, %.2fKb per frame" % (size_vid, size_vid / num_frames * 8), i / fps)
                up_bw_per_period.append(size_vid * 8)
                comp_cap = cv2.VideoCapture(output_video_file)
                comp_ret = True
                while comp_ret:
                    comp_ret, dec_frame = comp_cap.read()
                    if comp_ret:
                        dec_frame = cv2.resize(dec_frame, (SIZE[1], SIZE[0]))
                        dec_frame = cv2.cvtColor(dec_frame, cv2.COLOR_BGR2RGB)
                        frame_memory.append(dec_frame)
                os.remove(output_video_file)
            else:
                output_image_file = f"{get_save_dir(run_label)}_tmp_image.png"
                size_images = 0
                while len(to_compress_frame_memory) > 0:
                    f = to_compress_frame_memory.popleft()
                    cv2.imwrite(output_image_file, f)
                    size_images += os.path.getsize(output_image_file) / 1024
                    frame_memory.append(f)
                up_bw_per_period.append(size_images * 8)
                os.remove(output_image_file)

        if i // fps in save_range:
            if flags.enable_ASR:
                # Compute phi-score based on unseen frames and change send_rate
                i_start = max(0, len(label_memory) - num_unseen_frames - 1)
                miou_cross_arr_ = []
                for k in range(i_start, len(label_memory) - 1):
                    _, _, miou_cross_ = semantic_network.calc_cross_miou(
                        np.array([label_memory[k], label_memory[k + 1]]))
                    miou_cross_arr_.append(miou_cross_)
                send_rate = send_rate - 0.2 * np.tanh((np.mean(miou_cross_arr_) - 0.6) * 20)
                send_rate = np.clip(send_rate, 0.1, 1)
                print_process("Send rate updated to %.2f" % send_rate, i / fps)
                num_unseen_frames = 0

            if FLAGS.enable_ATR:
                # Enable or disable hibernation based on phi-score history
                if np.mean(list(send_rate_deq)) < 0.25:
                    hibernate = True
                if np.mean(list(send_rate_deq)) > 0.35 and hibernate:
                    hibernate = False
                    train_period_current = train_period_reset
                    print_process(f"Train period reset to {train_period_reset}", i/fps)
                if hibernate:
                    train_period_current = min(train_period_current+2, 6 * train_period_reset)
                    print_process(f"Train period updated to {train_period_current}", i/fps)
                # Change the next training times based on train_period_current
                index_now_save_range = save_range.index(i // fps)
                save_range = save_range[:index_now_save_range]
                save_range.extend([save_time for save_time in range(i//fps, train_end, train_period_current)])
                assert i // fps in save_range

            if not FLAGS.no_restore:
                semantic_network.restore_initial()
            t1 = time.time()
            semantic_network.train_with_deque(frame_memory, label_memory, FLAGS.iter, FLAGS.train_strategy)
            print("Training for %d iterations took %d ms!!!" % (FLAGS.iter, 1000 * (time.time() - t1)))
            whole_params = 0
            # Calculate the down-link bandwidth
            with open(save_dir + '_mask.dat', 'wb') as f:
                full_size = 0
                for val in semantic_network.curr_mask:
                    val_reshape = val.flatten()
                    whole_params += val_reshape.size
                    val_reshape = np.packbits(val_reshape)
                    f.write(val_reshape.tobytes())
                    full_size += val.size
                for p_ind in range(len(semantic_network.train_params)):
                    assert semantic_network.train_params[p_ind].shape == semantic_network.curr_mask[p_ind].shape
                    write_params = \
                        semantic_network.train_params[p_ind][semantic_network.curr_mask[p_ind]].astype(np.float16)
                    f.write(write_params.tobytes())
            # Experimental method: Add params to gzip as well, instead of sending changed params
            # if bw usage is worse switch to the version used for the paper
            sp.Popen(['gzip', '-9', '-f', '-k', save_dir + '_mask.dat']).wait()
            print("Full size of model is %d" % full_size)
            curr_update = os.path.getsize(save_dir + '_mask.dat.gz') * 8
            down_bw_per_period.append(curr_update)
            update_count += 1
            print("Using %.1fKbps for updating params" % (curr_update // 1024))
            # Save the model
            save_dir = get_save_dir(run_label + f"_{i // fps}")
            semantic_network.save_to_frozen_graph(save_dir + "_final")
            print_process("Saved model to %s_final.pb" % save_dir, i / fps)
            model_save_times.append(i / fps)

    semantic_network.close_model()
    final_save_dir = get_save_dir(run_label + "_results")
    np.save(final_save_dir + '_fps_client.npy', sample_per_period)
    np.save(final_save_dir + '_bw_uplink.npy', up_bw_per_period)
    np.save(final_save_dir + '_bw_downlink.npy', down_bw_per_period)
    np.save(final_save_dir + '_model_update_times.npy', model_save_times)
    # Write bandwidth stats
    with open(final_save_dir + '_update.txt', 'w') as f:
        interval = train_end - train_start
        if update_count == 0:
            assert len(down_bw_per_period) == 0
        downlink_size = sum(down_bw_per_period)
        uplink_size = sum(up_bw_per_period)
        samples_sent = sum(sample_per_period)
        f.write("%d\n%d\n%d\n%d\n%d" % (downlink_size, uplink_size, update_count, interval, samples_sent))
    cap.release()
    frame_memory.clear()
    label_memory.clear()
    to_compress_frame_memory.clear()


def infer_output(inf_start, inf_end, gpu_id, run_label, gt_path, exp_num, load_range):
    """
    This function emulates the client-side inference phase of the server-client setting. It infers frames  in the range
    [inf_start, inf_end). It loads models for times in load_range whenever it reaches them.

    :param inf_start: Start of the interval
    :param inf_end: End of the interval
    :param gpu_id: GPU index to use for the training
    :param run_label: A label used to recognize this experiment's output, must be unique to this experiment
    :param gt_path: Where ground truth labels are saved
    :param exp_num: A unique number assigned to each video that can be used to look up it's length and chosen classes
    :param load_range: The points to load new models
    :type inf_end: int
    :type inf_start: int
    :type gpu_id: str
    :type run_label: str
    :type gt_path: str
    :type exp_num: int
    :type load_range: list
    """
    assert inf_end - inf_start != 0, "There should be at least one set of data points"
    # Open video, get the fps and set it's starting point to train_start
    cap = cv2.VideoCapture(FLAGS.input_video)
    if not cap.isOpened():
        print_process("Error opening video stream or file", -1)
        exit(1)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    inf_end_frame = inf_end * fps
    i = inf_start * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)

    semantic_network = None
    confusion_matrix_memory = deque(maxlen=10 * fps)
    loss_s, miou_cats, miou_s, miou_mem_s = [], [], [], []
    final_save_dir = get_save_dir(run_label + "_results")

    while cap.isOpened() and i < inf_end_frame:
        if i / fps in load_range:
            # Load new model
            save_dir = get_save_dir(run_label + "_%d" % (i//fps))
            if semantic_network is not None:
                semantic_network.close_model()
            semantic_network = SemanticNetwork(meta_dir=save_dir + "_final",
                                               class_weights_exp=class_weights(exp_num),
                                               height=FLAGS.height,
                                               gpu_id=gpu_id,
                                               mem_frac=1,
                                               frozen=True)
        # Load frame, actual label, the model's prediction and compute mIoU and loss
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (SIZE[1], SIZE[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print("Premature end of video, exiting")
            exit(1)
        gt_frame = cv2.imread("%sgt_%06d.png" % (gt_path, i), cv2.IMREAD_GRAYSCALE)
        gt_frame = cv2.resize(gt_frame, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_NEAREST)
        labels_, conf_mat_, _, miou_, loss_ = semantic_network.predict_with_metric(np.expand_dims(frame, axis=0),
                                                                                   np.expand_dims(gt_frame, axis=0))
        loss_s.append(loss_)
        miou_cats.append(np.array(conf_mat_))
        miou_s.append(miou_)
        confusion_matrix_memory.append(conf_mat_)
        miou_mem_s.append(np.nanmean(calculate_miou(np.sum(list(confusion_matrix_memory), axis=0), nan=True)))
        i += 1
        # Log accuracy stats every second
        if i % fps == 0:
            miou = np.nanmean(calculate_miou(np.sum(miou_cats[-fps:], axis=0), nan=True))
            print_process("miou at %03d secs: %.1f%%" % (i / fps, float(miou)*100), i / fps)
            iou_class, pop_class, false_neg, false_pos = calculate_miou(np.sum(miou_cats[-fps:], axis=0),
                                                                        population=True, detailed=True)
            print_process("\n\n%s" % (string_class_iou([iou_class, false_neg, false_pos], population=pop_class,
                                                       headers=["Class IoU", "False Negative", "False Positive"],
                                                       class_weights=class_weights(exp_num))), i / fps)
        # Save visual results: the teachers output, the student's output, the ignored pixels and the student's
        # wrongly-predicted pixels
        if FLAGS.save_pic:
            save_dir_pic = final_save_dir + ("_%d_" % (i / fps))
            cross_mask, ignore_mask = semantic_network.cross_ignore(label_teacher=gt_frame,
                                                                    label_student=labels_[0])
            cv2.imwrite(save_dir_pic + "cross_mask.png", cv2.cvtColor(cross_mask, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_dir_pic + "ignore_mask.png", cv2.cvtColor(ignore_mask, cv2.COLOR_RGB2BGR))
            overlay_teacher, output_teacher = semantic_network.colorize_teacher(label=gt_frame, frame=frame)
            cv2.imwrite(save_dir_pic + "overlay_teacher.png", cv2.cvtColor(overlay_teacher, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_dir_pic + "output_teacher.png", cv2.cvtColor(output_teacher, cv2.COLOR_RGB2BGR))
            overlay_student, output_student = semantic_network.colorize(label=labels_[0], frame=frame)
            cv2.imwrite(save_dir_pic + "output_student.png", cv2.cvtColor(output_student, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_dir_pic + "overlay_student.png", cv2.cvtColor(overlay_student, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_dir_pic + "frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_dir_pic + "label_student.png", labels_[0])

    np.save('%s_loss.npy' % final_save_dir, loss_s)
    np.save('%s_mioucats.npy' % final_save_dir, miou_cats)
    np.save('%s_mious.npy' % final_save_dir, miou_s)
    np.save('%s_mioumems.npy' % final_save_dir, miou_mem_s)
    cap.release()
    semantic_network.close_model()


def k1k2_plot(ts, k1s, k2s):
    """
    This function calculates the relative improvement in average and confusion-matrix miou across different training
     (tau') and inference (tau) horizons. It is run after training and inference for each part.
    :param ts: Chosen time points
    :param k1s: The list of training horizons (tau')
    :param k2s: The list of inference horizons (tau)
    :type ts: list
    :type k1s: list
    :type k2s: list
    """
    cap = cv2.VideoCapture(FLAGS.input_video)
    if not cap.isOpened():
        print_process("Error opening video stream or file", -1)
        exit(1)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    # To compare against pretrained, first load data for the pretrained version
    # There are two methods to compute mIoU:
    # 1: sum up all the confusion matrices of frames and then calculate mIoU
    # 2: Calculate mIoU based on confusion matrices for each frame and average them
    # 3: Assign to each point t, the mIoU calculated from the sum of the confusion matrices of the last 10 seconds
    # Traditionally the first method is used in semantic segmentation literature but that is when we only use 1 model,
    # when multiple models are used, it would make sense to use the 2nd or 3rd version, to understand why this is, think
    # of an example where an initial model performs very badly but the 2nd model is perfect. The 1st version overstates
    # the bad performance at the start but the 2nd version gives these time spans the same weight. However the mIoU of
    # the individual frames are noisy. So we used the 3rd version which de-noises them a bit. Despite these
    # explanations, actual results showed all versions to follow the same trends and gaps.
    pretrained_confmats = np.load(get_save_dir("pretrained_results") + "_mioucats.npy")
    pretrained_mious = np.load(get_save_dir("pretrained_results") + "_mious.npy")
    pretrained_miou_mems = np.load(get_save_dir("pretrained_results") + "_mioumems.npy")
    results_conf_mious = np.empty((len(k1s), len(k2s)))
    results_avg_mious = np.empty((len(k1s), len(k2s)))
    results_miou_mems = np.empty((len(k1s), len(k2s)))
    for i_k1, k1 in enumerate(k1s):
        for i_k2, k2 in enumerate(k2s):
            diff_conf_mious, diff_avg_mious, diff_mem_mious = [], [], []
            for t in ts:
                trained_conf_mats = np.load(get_save_dir("%d__%d__%d_f%d_results" %
                                                         (t - k1, t, t + k2s[-1], FLAGS.send_period)) + "_mioucats.npy")
                assert trained_conf_mats[:k2 * fps].shape == pretrained_confmats[t * fps:(t + k2) * fps].shape
                pretrained_conf_miou = np.nanmean(calculate_miou(
                    np.sum(pretrained_confmats[t * fps:(t + k2) * fps], axis=0), nan=True))
                trained_conf_miou = np.nanmean(calculate_miou(np.sum(trained_conf_mats[:k2 * fps], axis=0), nan=True))
                diff_conf_mious.append(trained_conf_miou - pretrained_conf_miou)

                trained_mious = np.load(get_save_dir("%d__%d__%d_f%d_results" %
                                                     (t - k1, t, t + k2s[-1], FLAGS.send_period)) + "_mious.npy")
                assert trained_mious[:k2 * fps].shape == pretrained_mious[t * fps:(t + k2) * fps].shape
                pretrained_avg_miou = np.mean(pretrained_mious[t * fps:(t + k2) * fps])
                trained_avg_miou = np.mean(trained_mious[:k2 * fps])
                diff_avg_mious.append(trained_avg_miou - pretrained_avg_miou)

                trained_miou_mems = np.load(get_save_dir("%d__%d__%d_f%d_results" %
                                                     (t - k1, t, t + k2s[-1], FLAGS.send_period)) + "_mioumems.npy")
                assert trained_miou_mems[:k2 * fps].shape == pretrained_miou_mems[t * fps:(t + k2) * fps].shape
                pretrained_mioumem = np.mean(pretrained_miou_mems[t * fps:(t + k2) * fps])
                trained_mioumem = np.mean(trained_miou_mems[:k2 * fps])
                diff_mem_mious.append(trained_mioumem - pretrained_mioumem)
            results_conf_mious[i_k1, i_k2] = np.mean(diff_conf_mious)
            results_avg_mious[i_k1, i_k2] = np.mean(diff_avg_mious)
            results_miou_mems[i_k1, i_k2] = np.mean(diff_mem_mious)

    print("Confusions Matrix-Based mIoUs:")
    for i_k1, k1 in enumerate(k1s):
        for i_k2, k2 in enumerate(k2s):
            print(f'({k1}, {k2}, {results_conf_mious[i_k1, i_k2] * 100})')

    print("Average mIoUs:")
    for i_k1, k1 in enumerate(k1s):
        for i_k2, k2 in enumerate(k2s):
            print(f'({k1}, {k2}, {results_avg_mious[i_k1, i_k2] * 100})')

    print("Average mIoU memories:")
    for i_k1, k1 in enumerate(k1s):
        for i_k2, k2 in enumerate(k2s):
            print(f'({k1}, {k2}, {results_miou_mems[i_k1, i_k2] * 100})')


def plot_miou_mean(period, sampling_period, run_label):
    """
    This function prints the output of the experiment with this run_label.

    :param period: The retraining period
    :param sampling_period: The frame sampling rate
    :param run_label: The label of this experiment
    :type period: int
    :type sampling_period: int
    :type run_label: str
    """
    final_save_dir = get_save_dir(run_label + "_results")
    with open(final_save_dir + '_update.txt', 'r') as f:
        downlink_size, uplink_size, update_count, interval, samples_sent = [k.rstrip('\n') for k in f.readlines()]
    miou_s = np.load('%s_mioumems.npy' % get_save_dir(run_label + "_results"))
    print(f'({period}, {sampling_period}, {np.mean(miou_s[7500:]) * 100})')
    print(f'Uplink: {uplink_size / interval / 1024}, Downlink: {downlink_size / interval / 1024}, Sampling rate: '
          f'{samples_sent / interval}, Update rate: {update_count / interval}')


def get_save_dir(prepend):
    """
    This helper function returns a label, given a prepending string and the arguments.
    :param prepend: A string starting the experiments filenames
    :type prepend: str
    :return: a label unique to that experiment
    :rtype: str
    """

    return FLAGS.output_dir + '%s_%s_%s_%d' % (prepend, FLAGS.input_video.split('/')[-1],
                                               FLAGS.student_checkpoint.split('/')[-2], FLAGS.height)


def print_process(str_log, curr_time):
    """
    This helper function tidies up command line outputs.
    :param str_log: a string to print
    :param curr_time: The time in the experiment
    """
    print(colored('Process [current time: %d]: ' % curr_time, 'cyan'), str_log)


def main():
    try:
        os.makedirs(FLAGS.output_dir)
    except FileExistsError:
        pass

    vid_num = int(FLAGS.input_video.split("/")[-1].split("-")[0])

    if FLAGS.mode == 'simple':
        run_label = "%d__%d_tp%d_f%d" % (0, test_length(vid_num), FLAGS.train_period, FLAGS.send_period)
        event_list = [0]
        first_train = np.ceil(100 / FLAGS.train_period) * FLAGS.train_period
        event_list.extend([i for i in range(first_train, test_length(vid_num), FLAGS.train_period)
                           if i == 0 or i >= FLAGS.memory_len or not FLAGS.initial_fill])
        if not FLAGS.only_results:
            train_model(0, test_length(vid_num), FLAGS.send_period, FLAGS.gpu,
                        run_label, FLAGS.gt_video, vid_num, event_list, FLAGS.train_period)
            if flags.enable_ATR:
                # When ATR is enabled event_list can change, so load it
                event_list = np.load(get_save_dir(run_label + "_results") + '_model_update_times.npy').tolist()
            infer_output(0, test_length(vid_num), FLAGS.gpu,
                         run_label, FLAGS.gt_video, vid_num, event_list)

        plot_miou_mean(FLAGS.train_period, FLAGS.send_period, run_label)
    elif FLAGS.mode == 'horizon':
        # TODO: Arash, please double check this part to make sure it is exactly what we did for the paper, especially
        # TODO: the hyper-parameters
        k1s = [16, 32, 64, 128, 256, 512]
        k2 = 256
        # Choose 50 points across time to smooth out noisy curves
        number_of_points = 3
        step = (test_length(vid_num) - k2 - k1s[-1]) // (number_of_points - 1)
        if not FLAGS.only_results:
            # Get pretrained data
            run_label = "pretrained"
            train_model(0, 1, FLAGS.send_period, FLAGS.gpu, run_label, FLAGS.gt_video, vid_num, [0], FLAGS.train_period)
            infer_output(0, test_length(vid_num), FLAGS.gpu, run_label, FLAGS.gt_video, vid_num, [0])
            done = 0
            total = number_of_points * len(k1s)
            time_start = time.time()
            for i in range(number_of_points):
                t = k1s[-1] + i * step
                for k1 in k1s:
                    run_label = "%d__%d__%d_f%d" % (t - k1, t, t + k2, FLAGS.send_period)
                    print("t: %d, k1: %d" % (t, k1))
                    train_model(t - k1, t, FLAGS.send_period, FLAGS.gpu, run_label, FLAGS.gt_video, vid_num, [t],
                                FLAGS.train_period)
                    infer_output(t, t + k2, FLAGS.gpu, run_label, FLAGS.gt_video, vid_num, [t])
                    done += 1
                    time_to_finish = (time.time() - time_start) / done * (total - done)
                    print("ETF %02d:%02d.%02d" % (time_to_finish // 60, time_to_finish % 60,
                                                  (time_to_finish * 100) % 100))

        k2s = [16, 32, 64, 128, 256]
        ts = []
        for i in range(number_of_points):
            t = k1s[-1] + i * step
            ts.append(t)
        k1k2_plot(ts, k1s, k2s)
    elif FLAGS.mode == 'early':
        run_label = "early%d_f%d" % (FLAGS.early_cutoff_time, FLAGS.send_period)
        event_list = [0, FLAGS.early_cutoff_time]
        if not FLAGS.only_results:
            train_model(0, FLAGS.early_cutoff_time, FLAGS.send_period, FLAGS.gpu, run_label, FLAGS.gt_video, vid_num,
                        event_list, FLAGS.train_period)
            infer_output(0, test_length(vid_num), FLAGS.gpu, run_label, FLAGS.gt_video, vid_num, event_list)

        plot_miou_mean(-1, FLAGS.send_period, run_label)
    elif FLAGS.mode == 'pretrained':
        run_label = "pretrained"
        train_model(0, 1, FLAGS.send_period, FLAGS.gpu, run_label, FLAGS.gt_video, vid_num, [0], FLAGS.train_period)
        infer_output(0, test_length(vid_num), FLAGS.gpu, run_label, FLAGS.gt_video, vid_num, [0])
        plot_miou_mean(-1, -1, run_label)

    print(colored("Process [Main]:", "green"), "Done!!!")


if __name__ == "__main__":
    main()
