import os
import threading
import time

import numpy as np
import sys
import cv2
from collections import deque

from termcolor import colored

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

sys.path.append('../../.')
from ams.utils.graph_utils import create_student_v3, trim_graph_frozen
from ams.utils.utils import SaveHelper, calculate_miou, colormap, mini_batch

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# TODO: simplify code, remove the sys.append, test running it, merge with other profilers


class SemanticNetwork(object):
    OPT_FILTER = ['Adam', 'Momentum']
    OP_FILTER = ['image_cache:0', 'global_step:0']
    THREAD_SLEEP_INTERVAL = 1 / 1000.
    TOTAL_CLASSES = 19
    WHITE = np.array([255, 255, 255], dtype=np.uint8)
    BLACK = np.array([0, 0, 0], dtype=np.uint8)

    def __init__(self, meta_dir, class_weights_exp=None, height=None, gpu_id='0', frozen=False,
                 scale=None, mini_batch_size=None, lr=None, mem_frac=1, coord_frac=0.1, cross_miou_compat=False,
                 filter_out=None, over_ride_total_classes=None, **kwargs):
        assert height is not None, "No height is given"
        assert class_weights_exp is not None, "No class weights specified"
        assert frozen or None not in [scale, mini_batch_size, lr], "Training parameters must be specified for " \
                                                                   "non-frozen graph"
        self.lr = lr
        self.mini_batch_size = mini_batch_size
        self.scale = scale
        if over_ride_total_classes is not None:
            print(colored('Overriding default number of classes', 'cyan'))
            self.TOTAL_CLASSES = over_ride_total_classes

        self.coord_frac = coord_frac

        self.class_weights_graph = class_weights_exp
        self.class_indices_graph = np.where(self.class_weights_graph == 1)[0]
        assert self.class_weights_graph.shape == (self.TOTAL_CLASSES, 1)
        self.class_count = len(self.class_indices_graph)
        assert self.class_indices_graph.shape == (self.class_count,)
        assert self.class_count > 0

        self.cross_miou_compat = cross_miou_compat

        self.color_map_reduced_ = np.take(colormap(), self.class_indices_graph, axis=0)
        self.take_array = np.cumsum(self.class_weights_graph).reshape(
            self.TOTAL_CLASSES) * self.class_weights_graph.reshape(self.TOTAL_CLASSES)
        self.take_array = np.where(self.take_array != 0, self.take_array - 1, self.take_array)
        self.take_array = self.take_array.astype(int)
        assert self.take_array.shape == (self.TOTAL_CLASSES,)

        self.frozen = frozen
        self.height = height
        assert self.height > 0

        self.meta_dir = meta_dir

        self.process_lock = threading.Lock()
        self.config = tf.ConfigProto()
        if mem_frac != 1:
            self.config.gpu_options.per_process_gpu_memory_fraction = mem_frac
        self.config.gpu_options.visible_device_list = gpu_id
        self.config.allow_soft_placement = True
        self.config.gpu_options.allow_growth = False

        tf.reset_default_graph()

        if self.frozen:
            graph_def = tf.GraphDef()
            with open(meta_dir + ".pb", 'rb') as pb_file:
                graph_def.ParseFromString(pb_file.read())

            graph = tf.Graph()

            with tf.device('/gpu:0'):
                with graph.as_default():
                    self.frozen_predictions = tf.import_graph_def(graph_def, return_elements=['student_predictions:0'],
                                                                  name='')[0]
                    self.frozen_logits = graph.get_tensor_by_name('logits_reduced:0')
                    self.frozen_image = graph.get_tensor_by_name('features:0')
                    self.frozen_image.set_shape([1, self.height, self.height * 2, 3])

                    init = tf.initializers.global_variables()
                    self.frozen_labels_pl = tf.placeholder(tf.int32, shape=[None, None, None])
                    labels_onehot = tf.one_hot(self.frozen_labels_pl, 19, axis=-1)
                    filtered_labels_onehot = tf.gather(labels_onehot, self.class_indices_graph, axis=-1)
                    filtered_labels = tf.argmax(filtered_labels_onehot, axis=-1)
                    weights = tf.reduce_sum(filtered_labels_onehot, axis=-1)

                    self.mean_iou, self.update_op = tf.metrics.mean_iou(
                        labels=filtered_labels,
                        predictions=self.frozen_predictions,
                        num_classes=self.class_count,
                        weights=weights)
                    miou_list_vars = [v for v in tf.local_variables() if any(tag in v.name for tag in
                                                                             ['confusion', 'miou', 'mean_iou'])]
                    self.reset_conf_mat = tf.variables_initializer(miou_list_vars)

                    pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.frozen_logits,
                                                                         labels=filtered_labels_onehot)

                    weights = tf.cast(weights, tf.bool)
                    self.loss = tf.reduce_mean(tf.boolean_mask(pixel_loss, weights))

            self.sess = tf.Session(config=self.config, graph=graph)
            self.sess.run([init, self.reset_conf_mat])
        else:
            with tf.device('/gpu:0'):
                self.student = create_student_v3(meta_dir, class_weights=class_weights_exp, **kwargs)
                self.saver = SaveHelper(graph=self.student['graph'], map_fun=lambda x: x)
                with self.student['graph'].as_default():
                    if cross_miou_compat:
                        self.labels_after = tf.placeholder(tf.int32, [None, None])
                        labels_after_one_hot = tf.one_hot(self.labels_after, self.TOTAL_CLASSES, axis=-1)
                        labels_after_one_hot_reduced = tf.gather(labels_after_one_hot, self.class_indices_graph, axis=-1)
                        labels_after_reduced = tf.argmax(labels_after_one_hot_reduced, axis=-1, output_type=tf.int32)
                        self.labels_before = tf.placeholder(tf.int32, [None, None])
                        labels_before_one_hot = tf.one_hot(self.labels_before, self.TOTAL_CLASSES, axis=-1)
                        labels_before_one_hot_reduced = tf.gather(labels_before_one_hot, self.class_indices_graph, axis=-1)
                        labels_before_reduced = tf.argmax(labels_before_one_hot_reduced, axis=-1, output_type=tf.int32)
                        weights = tf.reduce_max(labels_before_one_hot_reduced, axis=-1) * \
                                  tf.reduce_max(labels_after_one_hot_reduced, axis=-1)
                        self.cross_mean_iou, self.cross_update_op = tf.metrics.mean_iou(
                            labels=labels_before_reduced,
                            predictions=labels_after_reduced,
                            num_classes=self.class_count,
                            weights=weights)

                    miou_list_vars = [v for v in tf.local_variables() if any(tag in v.name for tag in
                                                                             ['confusion', 'miou', 'mean_iou'])]
                    self.reset_conf_mat = tf.variables_initializer(miou_list_vars)
                    init = tf.initializers.global_variables()

                    self.save_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name not in
                                      ['images:0', 'labels:0', 'label_cache:0', 'image_cache:0']]

            self.sess = tf.Session(graph=self.student['graph'], config=self.config)
            self.sess.run([init, self.reset_conf_mat])

            if filter_out is not None:
                self.OPT_FILTER.extend(filter_out)
            self.filter = lambda elem: elem if all(
                keyword not in elem for keyword in self.OPT_FILTER) and elem not in self.OP_FILTER else None
            self.saver.restore_vars(self.sess, "%s.npy" % self.meta_dir, self.filter)
            self.mask = None

        print("Semantic Network is ready!!!")

    def restore_initial(self):
        self.saver.restore_vars(self.sess, "%s.npy" % self.meta_dir, self.filter)

    def restore(self, chk):
        self.saver.restore_vars(self.sess, chk, self.filter)

    def get_vars(self):
        return self.saver.save_vars(self.sess, self.save_vars, lambda x: x)

    def predict_input(self, frames):
        self.process_lock.acquire()
        if self.frozen:
            labels_ = self.sess.run(self.frozen_predictions,
                                    feed_dict={self.frozen_image: frames})
        else:
            self.sess.run(self.student['fill_input_buffer'],
                          feed_dict={self.student['features_input']: frames,
                                     self.student['labels_input']: np.zeros((frames.shape[:-1]))})
            labels_ = self.sess.run(self.student['predictions'])
        assert labels_.shape == frames.shape[:-1]
        self.process_lock.release()
        return labels_

    def calc_cross_miou(self, labels):
        assert not self.frozen or self.cross_miou_compat
        assert labels.shape == (2, self.height, 2 * self.height)
        self.process_lock.acquire()
        self.sess.run(self.reset_conf_mat)
        conf_mat_ = self.sess.run(self.cross_update_op, feed_dict={self.labels_before: labels[0],
                                                                   self.labels_after: labels[1]})
        iou_ = calculate_miou(conf_mat_, nan=True)
        miou_ = np.nanmean(iou_)
        self.process_lock.release()
        return conf_mat_, iou_, miou_

    def predict_with_metric(self, frames, labels_teacher):
        self.process_lock.acquire()
        self.sess.run(self.reset_conf_mat)
        if self.frozen:
            labels_student, conf_mat_, loss_ = self.sess.run([self.frozen_predictions, self.update_op, self.loss],
                                                             feed_dict={self.frozen_labels_pl: labels_teacher,
                                                                        self.frozen_image: frames})
        else:
            self.sess.run(self.student['fill_input_buffer'],
                          feed_dict={self.student['features_input']: frames,
                                     self.student['labels_input']: labels_teacher})
            labels_student, conf_mat_, loss_ = self.sess.run([self.student['predictions'], self.student['update_op'],
                                                              self.student['loss']])
        assert labels_student.shape == frames.shape[:-1]
        iou_ = calculate_miou(conf_mat_, nan=True)
        miou_ = np.nanmean(iou_)
        self.process_lock.release()
        return labels_student, conf_mat_, iou_, miou_, loss_

    def train_with_deque(self, frame_deque, label_deque, num_of_iterations, train_strategy='full_model',
                         keep_mask=False):
        assert not self.frozen, "Can't train frozen graph!!!"
        if not keep_mask:
            self.mask = None
        self.process_lock.acquire()
        batch_deque = deque()
        batch_thr = threading.Thread(target=self._fill_batch, args=(batch_deque, frame_deque, label_deque,
                                                                    num_of_iterations,))
        batch_thr.start()
        self._train(batch_deque, num_of_iterations, train_strategy)

    def _train(self, batch_deque, num_of_iterations, train_strategy):

        signal_deque = deque()
        fill_thr = threading.Thread(target=self._fill_queue, args=(batch_deque, num_of_iterations, signal_deque))
        fill_thr.start()

        if 'coord_desc_' in train_strategy:
            train_node = self.student['train_coord']
        else:
            train_node = self.student['train']

        _before, train_mask_ = self.get_train_mask(train_strategy)

        iteration_update_ops = {'train_node': train_node,
                                'loss': self.student['loss']}

        for it in range(num_of_iterations):
            signal = None
            while signal is None:
                try:
                    signal = signal_deque.popleft()
                except IndexError:
                    time.sleep(self.THREAD_SLEEP_INTERVAL)
            t1 = time.time()

            # Construct the feed_dict
            feed_dict = {self.student['learning_rate']: self.lr}

            if 'coord_desc_' in train_strategy:
                for k in train_mask_:
                    feed_dict[k] = train_mask_[k]

            # Call for execution
            results = self.sess.run(iteration_update_ops, feed_dict=feed_dict)
            print('Loss is %.3f at iteration %d and took %.1f ms' % (results['loss'], it, (time.time() - t1) * 1000.0))

            if train_strategy == 'coord_desc_auto':
                if it == 0 and self.mask is None:
                    # Update the train_mask
                    _after = self.saver.save_vars(self.sess, self.save_vars, self.filter)
                    changes = []
                    for k in self.student['grad_masks_pl']:
                        changes.append(np.reshape(np.abs(_after[k] - _before[k]), (-1,)))
                    changes = np.concatenate(changes, axis=0)
                    cut_threshold = np.percentile(changes, 100*(1-self.coord_frac))
                    numvars_list = []
                    train_vars_len = 0
                    all_vars = 0
                    _combine = {}
                    for var_name in self.student['grad_masks_pl']:
                        train_mask_[self.student['grad_masks_pl'][var_name]] = np.abs(
                            _after[var_name] - _before[var_name]) > cut_threshold

                        train_vars_len += np.sum(train_mask_[self.student['grad_masks_pl'][var_name]])
                        all_vars += train_mask_[self.student['grad_masks_pl'][var_name]].size
                        numvars_list.append(np.sum(train_mask_[self.student['grad_masks_pl'][var_name]]))
                        _combine[var_name] = np.where(train_mask_[self.student['grad_masks_pl'][var_name]],
                                                      _after[var_name], _before[var_name])
                        assert _combine[var_name].shape == _before[var_name].shape
                    print("Using auto mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
                    self.saver.restore_vars(self.sess, _combine, self.filter)
                    self.mask = train_mask_

        if 'coord_desc_' in train_strategy:
            self.curr_mask = [train_mask_[self.student['grad_masks_pl'][var_name]]
                              for var_name in self.student['grad_masks_pl']]
            _after_train = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            self.train_params = [_after_train[var_name] for var_name in self.student['grad_masks_pl']]
        else:
            _after_train = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            self.train_params = [_after_train[var_name] for var_name in _after_train.keys()]
            self.curr_mask = [np.ones_like(_after_train[var_name], dtype=np.bool) for var_name in _after_train.keys()]

        self.process_lock.release()

    def get_train_mask(self, train_strategy):
        if train_strategy == 'coord_desc_auto':
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            if self.mask is None:
                train_mask_ = {self.student['grad_masks_pl'][var_name]: np.ones(_before[var_name].shape, dtype=np.bool)
                               for var_name in self.student['grad_masks_pl']}
            else:
                train_mask_ = self.mask
        elif train_strategy == 'coord_desc_last' and self.coord_frac == 0.1:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in ['aspp0/BatchNorm/gamma:0', 'aspp0/BatchNorm/beta:0', 'concat_projection/weights:0',
                        'concat_projection/BatchNorm/gamma:0', 'concat_projection/BatchNorm/beta:0',
                        'logits/semantic/weights:0', 'logits/semantic/biases:0']:
                assert self.student['grad_masks_pl'][key] in train_mask_
                train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
            key = 'aspp0/weights:0'
            assert self.student['grad_masks_pl'][key] in train_mask_
            train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False], size=(1, 1, 320, 256),
                                                                               p=[0.90728, 0.09272]).astype(np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using last10 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))

        elif train_strategy == 'coord_desc_first' and self.coord_frac == 0.1:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/',
                                                      '/expanded_conv_4/',
                                                      '/expanded_conv_5/',
                                                      '/expanded_conv_6/',
                                                      '/expanded_conv_7/',
                                                      '/expanded_conv_8/']) or \
                        (key in ['MobilenetV2/expanded_conv_9/expand/weights:0',
                                 'MobilenetV2/expanded_conv_9/expand/BatchNorm/gamma:0',
                                 'MobilenetV2/expanded_conv_9/expand/BatchNorm/beta:0']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_9/depthwise/depthwise_weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.25231, 0.74769]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using first10 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_both' and self.coord_frac == 0.1:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/',
                                                      '/expanded_conv_4/',
                                                      '/expanded_conv_5/',
                                                      '/expanded_conv_6/',
                                                      'logits/semantic/']) or \
                        (key in ['MobilenetV2/expanded_conv_7/expand/weights:0',
                                 'MobilenetV2/expanded_conv_7/expand/BatchNorm/gamma:0',
                                 'MobilenetV2/expanded_conv_7/expand/BatchNorm/beta:0',
                                 'MobilenetV2/expanded_conv_7/depthwise/depthwise_weights:0',
                                 'concat_projection/BatchNorm/gamma:0',
                                 'concat_projection/BatchNorm/beta:0']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_7/depthwise/BatchNorm/gamma:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.80208, 0.19792]).astype(
                        np.bool)
                elif key == 'concat_projection/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.76490, 0.23510]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using both10 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_last' and self.coord_frac == 0.05:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if 'logits/semantic/' in key or \
                        (key in ['concat_projection/BatchNorm/gamma:0',
                                 'concat_projection/BatchNorm/beta:0']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'concat_projection/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.76490, 0.23510]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using last5 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_first' and self.coord_frac == 0.05:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/',
                                                      '/expanded_conv_4/',
                                                      '/expanded_conv_5/',
                                                      '/expanded_conv_6/']) or \
                        (key in ['MobilenetV2/expanded_conv_7/expand/weights:0',
                                 'MobilenetV2/expanded_conv_7/expand/BatchNorm/gamma:0',
                                 'MobilenetV2/expanded_conv_7/expand/BatchNorm/beta:0',
                                 'MobilenetV2/expanded_conv_7/depthwise/depthwise_weights:0']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_7/depthwise/BatchNorm/gamma:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.80208, 0.19792]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using first5 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_both' and self.coord_frac == 0.05:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/',
                                                      '/expanded_conv_4/',
                                                      '/expanded_conv_5/expand/',
                                                      '/expanded_conv_5/depthwise/',
                                                      'logits/semantic/']) or \
                        (key in ['concat_projection/BatchNorm/gamma:0',
                                 'concat_projection/BatchNorm/beta:0']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_5/project/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.42285, 0.57715]).astype(
                        np.bool)
                elif key == 'concat_projection/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.36187, 0.63813]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using both5 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_last' and self.coord_frac == 0.01:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['logits/semantic/',
                                                      'concat_projection/BatchNorm/']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'concat_projection/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.12005, 0.87995]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using last1 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))

        elif train_strategy == 'coord_desc_first' and self.coord_frac == 0.01:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/depthwise/',
                                                      '/expanded_conv_3/expand/']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_3/project/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.00217, 0.99783]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using first1 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_both' and self.coord_frac == 0.01:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      'logits/semantic/',
                                                      'concat_projection/BatchNorm/']) or \
                        (key in ['MobilenetV2/expanded_conv_2/expand/weights:0',
                                 'MobilenetV2/expanded_conv_2/expand/BatchNorm/gamma:0']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_2/expand/BatchNorm/beta:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.03472, 0.96528]).astype(
                        np.bool)
                elif key == 'concat_projection/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.03944, 0.96056]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using both1 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_last' and self.coord_frac == 0.2:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['logits/semantic/',
                                                      'concat_projection/',
                                                      'aspp0/',
                                                      'image_pooling/',
                                                      'MobilenetV2/expanded_conv_16/project/BatchNorm']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_16/project/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.39270, 0.60730]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using last20 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_first' and self.coord_frac == 0.2:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/',
                                                      '/expanded_conv_4/',
                                                      '/expanded_conv_5/',
                                                      '/expanded_conv_6/',
                                                      '/expanded_conv_7/',
                                                      '/expanded_conv_8/',
                                                      '/expanded_conv_9/',
                                                      '/expanded_conv_10/',
                                                      '/expanded_conv_11/expand/',
                                                      '/expanded_conv_11/depthwise/']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_11/project/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.97367, 0.02633]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using first20 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_both' and self.coord_frac == 0.2:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/',
                                                      '/expanded_conv_4/',
                                                      '/expanded_conv_5/',
                                                      '/expanded_conv_6/',
                                                      '/expanded_conv_7/',
                                                      '/expanded_conv_8/',
                                                      'concat_projection/',
                                                      'aspp0/BatchNorm/',
                                                      'logits/semantic/']) or \
                        (key in ['MobilenetV2/expanded_conv_9/expand/weights:0',
                                 'MobilenetV2/expanded_conv_9/expand/BatchNorm/gamma:0',
                                 'MobilenetV2/expanded_conv_9/expand/BatchNorm/beta:0']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_9/depthwise/depthwise_weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.25231, 0.74769]).astype(
                        np.bool)
                elif key == 'aspp0/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.90728, 0.09272]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using both20 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_last' and self.coord_frac == 0.02:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['logits/semantic/',
                                                      'concat_projection/BatchNorm/']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'concat_projection/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.7187, 0.2813]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using last2 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_first' and self.coord_frac == 0.02:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/',
                                                      '/expanded_conv_4/']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_5/expand/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.7367, 0.2633]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using first2 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_both' and self.coord_frac == 0.02:
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {self.student['grad_masks_pl'][var_name]: np.zeros(_before[var_name].shape, dtype=np.bool)
                           for var_name in self.student['grad_masks_pl']}
            for key in self.student['grad_masks_pl']:
                if any(keyword in key for keyword in ['/Conv/',
                                                      '/expanded_conv/',
                                                      '/expanded_conv_1/',
                                                      '/expanded_conv_2/',
                                                      '/expanded_conv_3/depthwise/',
                                                      '/expanded_conv_3/expand/',
                                                      'logits/semantic/',
                                                      'concat_projection/BatchNorm/']):
                    train_mask_[self.student['grad_masks_pl'][key]] = np.ones(_before[key].shape, dtype=np.bool)
                elif key == 'MobilenetV2/expanded_conv_3/project/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.00217, 0.99783]).astype(
                        np.bool)
                elif key == 'concat_projection/weights:0':
                    train_mask_[self.student['grad_masks_pl'][key]] = np.random.choice([True, False],
                                                                                       size=_before[key].shape,
                                                                                       p=[0.12005, 0.87995]).astype(
                        np.bool)
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using both2 mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'coord_desc_rand':
            _before = self.saver.save_vars(self.sess, self.save_vars, self.filter)
            train_mask_ = {
                self.student['grad_masks_pl'][var_name]: np.random.choice([True, False], size=_before[var_name].shape,
                                                                          p=[self.coord_frac,
                                                                             1-self.coord_frac]).astype(np.bool)
                for var_name in self.student['grad_masks_pl']}
            all_vars, train_vars_len = self.train_vars_count(train_mask_)
            print("Using rand mode, Training %.3f%% of variables" % (100 * train_vars_len / all_vars))
        elif train_strategy == 'full_model':
            _before = None
            train_mask_ = None
        else:
            raise NameError('train_strategy %s is not implemented.' % train_strategy)

        return _before, train_mask_

    def train_vars_count(self, train_mask_):
        all_vars = 0
        train_vars_len = 0
        for var_name in self.student['grad_masks_pl']:
            train_vars_len += np.sum(train_mask_[self.student['grad_masks_pl'][var_name]])
            all_vars += train_mask_[self.student['grad_masks_pl'][var_name]].size
        return all_vars, train_vars_len

    def _fill_batch(self, batch_deque, frame_deque, label_deque, number_of_batches):
        for batch_index in range(number_of_batches):
            image_batch, label_batch = mini_batch(frame_deque,
                                                  label_deque,
                                                  [self.height, self.height * 2],
                                                  self.scale,
                                                  self.mini_batch_size,
                                                  1,
                                                  flip=False)

            assert np.shape(label_batch) == (1, self.mini_batch_size, self.height, self.height * 2)
            assert np.shape(image_batch) == (1, self.mini_batch_size, self.height, self.height * 2, 3)
            batch_deque.append({'frames': image_batch[0], 'labels': label_batch[0]})

    def _fill_queue(self, batch_deque, number_of_batches, signal_deque):
        for batch_index in range(number_of_batches):
            batch = None
            while batch is None:
                try:
                    batch = batch_deque.popleft()
                except IndexError:
                    time.sleep(self.THREAD_SLEEP_INTERVAL)
            self.sess.run(self.student['fill_input_buffer'],
                          feed_dict={self.student['features_input']: batch['frames'],
                                     self.student['labels_input']: batch['labels']})
            signal_deque.append(1)

    def get_frozen_graph(self):
        graph_def = self.sess.graph_def
        return trim_graph_frozen(self.sess, graph_def, ["features"], [self.student["prepend"] + "predictions"],
                                 kill_norms=True)

    def save_to_frozen_graph(self, save_dir):
        graph_def = self.get_frozen_graph()
        with open(save_dir + ".pb", 'wb') as pb_file:
            pb_file.write(graph_def.SerializeToString())

    def close_model(self):
        self.sess.close()

    def colorize(self, frame=None, label=None):
        assert frame is not None or label is not None, "At least a label or frame must be given"
        assert frame is None or frame.shape == (self.height, self.height * 2, 3)
        if label is None:
            label = self.predict_input(np.expand_dims(frame, axis=0))[0]
        assert label.shape == (self.height, self.height * 2)
        label_colored = self.color_map_reduced_[label]
        if frame is not None:
            return label_colored, cv2.addWeighted(frame, 0.5, label_colored, 0.5, 0)
        else:
            return label_colored

    def colorize_teacher(self, label, frame=None):
        assert frame is None or frame.shape == (self.height, self.height * 2, 3)
        assert label.shape == (self.height, self.height * 2)
        label_colored = colormap()[label]
        if frame is not None:
            return label_colored, cv2.addWeighted(frame, 0.5, label_colored, 0.5, 0)
        else:
            return label_colored

    def cross_ignore(self, label_teacher, label_student=None, frame_student=None):
        assert label_student is not None or frame_student is not None, \
            "At least a label or frame from student must be given"
        assert label_teacher.shape == (self.height, self.height * 2)
        label_teacher_reduced = self.take_array[label_teacher]
        if label_student is None:
            label_student = self.predict_input(np.expand_dims(frame_student, axis=0))[0]
        assert label_student.shape == (self.height, self.height * 2)
        ignore_mask = np.where(np.expand_dims(label_teacher_reduced, axis=-1) == 0, self.WHITE, self.BLACK)
        colorized_label_teacher = self.colorize(label=label_teacher_reduced)
        cross_cond = np.logical_and(np.logical_not(ignore_mask[:, :, :1]),
                                    np.expand_dims(np.not_equal(label_teacher_reduced, label_student), axis=-1))
        cross_mask = np.where(cross_cond, colorized_label_teacher, self.BLACK)
        assert ignore_mask.shape == cross_mask.shape
        assert ignore_mask.shape == (self.height, self.height * 2, 3)
        return cross_mask, ignore_mask
