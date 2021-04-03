import os
import sys
import time
import numpy as np
import cv2
from termcolor import colored

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

sys.path.append('../../.')

from ams.tools.exp_configs import class_weights, test_length
from ams.utils.graph_utils import create_teacher
from ams.utils.utils import SaveHelper, colormap

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dump_path', None, 'Directory of the path data')
flags.DEFINE_string('teacher_checkpoint', None, 'Directory for teacher checkpoint')
flags.DEFINE_integer('gpu', 0, 'GPU to use for this')
flags.DEFINE_string('input_video', None, "Video used in the test, optional")
flags.DEFINE_integer('height', None, 'height to extract labels')

NUM_CLASSES = 19

# TODO: simplify code, remove the sys.append, test running it.


def extract_labels():
    try:
        os.makedirs(FLAGS.dump_path)
    except FileExistsError:
        pass

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "%d" % FLAGS.gpu
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = False

    class_weights_exp = class_weights(0)

    colormap_ = colormap()
    exp_num = int(FLAGS.input_video.split("/")[-1].split("-")[0])

    with tf.device('/gpu:0'):
        graph = tf.Graph()
        with graph.as_default():
            teacher = create_teacher(FLAGS.teacher_checkpoint, class_weights=class_weights_exp, test_mode=True)
            reset_conf_mat = tf.initializers.local_variables()
            init = tf.initializers.global_variables()
    saver = SaveHelper(graph=graph, map_fun=lambda x: x)
    with tf.Session(graph=graph, config=config) as sess:
        print("Starting Teacher Inference")
        sess.run([init, reset_conf_mat])
        teacher_checkpoint = np.load('%s.npy' % FLAGS.teacher_checkpoint, allow_pickle=True).item()
        teacher_checkpoint = {'teacher/%s' % k: teacher_checkpoint[k] for k in teacher_checkpoint}
        saver.restore_vars(sess, teacher_checkpoint,
                           lambda x: x if x not in ['global_step:0'] and 'Momentum' not in x else None)

        cap = cv2.VideoCapture(FLAGS.input_video)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        max_length = test_length(exp_num) * fps
        if cap.isOpened() is False:
            print(colored("Error opening video stream or file", "red"))
            return
        print("There are %d frames to extract" % max_length)

        index_frame = 0
        if not os.path.exists(FLAGS.dump_path):
            os.makedirs(FLAGS.dump_path)

        begin_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or index_frame >= max_length:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if FLAGS.height is not None:
                frame = cv2.resize(frame, (FLAGS.height * 2, FLAGS.height))
            correct_shape = np.shape(frame)[:-1]
            frame = np.pad(frame, ((1, 0), (1, 0), (0, 0)), mode='symmetric')
            teacher_out = sess.run(teacher['predictions'], feed_dict={teacher['images']: np.expand_dims(frame, axis=0)})
            teacher_out = [teacher_out[0][1:, 1:]]
            teacher_conf = teacher_conf[0][1:, 1:] * 255
            assert np.shape(teacher_out[0]) == correct_shape
            assert np.shape(teacher_conf) == correct_shape
            cv2.imwrite("%sgt_%06d.png" % (FLAGS.dump_path, index_frame), np.array(teacher_out[0], dtype=np.uint8))

            label_colored = colormap_[teacher_out[0]]
            cv2.imwrite("%sannot_%06d.png" % (FLAGS.dump_path, index_frame),
                        cv2.cvtColor(np.array(label_colored, dtype=np.uint8), cv2.COLOR_RGB2BGR))
            colored_frame = cv2.addWeighted(np.array(frame[1:, 1:, :], dtype=np.uint8), 0.5,
                                            np.array(label_colored, dtype=np.uint8), 0.5, 0)
            cv2.imwrite("%svis_%06d.png" % (FLAGS.dump_path, index_frame),
                        cv2.cvtColor(np.array(colored_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))

            index_frame += 1
            if index_frame % 100 == 0:
                time_to_finish = (time.time() - begin_time) / index_frame * (max_length - index_frame)
                print('Have computed %d frames so far, ETF: %02d:%02d.%02d' % (index_frame, time_to_finish // 60,
                                                                               time_to_finish % 60,
                                                                               (time_to_finish * 100) % 100))


if __name__ == "__main__":
    print("Extracting labels...")
    extract_labels()
