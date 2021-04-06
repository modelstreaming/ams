import numpy as np
import tensorflow as tf
import cv2
from collections import deque
import random

# TODO: simplify code, remove the sys.append, test running it, merge with other profilers


class SaveHelper:
    def __init__(self, graph, map_fun):
        with graph.as_default():
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.vars_dict = {map_fun(v.name): v for v in variables}
            self.vars_pl = {
                map_fun(v.name): tf.placeholder(dtype=v.dtype, shape=v.shape, name='pl_%s' % (v.name.strip(':0'))) for v
                in variables}
            self.load_ops = {k: tf.assign(self.vars_dict[k], self.vars_pl[k], use_locking=True) for k in self.vars_dict}

    def save_vars(self, sess, vars_list, map_fun, save_dir=None):
        vars_vals = sess.run(vars_list)
        save_dict = {}
        for i in range(len(vars_list)):
            if map_fun(vars_list[i].name):
                save_dict[map_fun(vars_list[i].name)] = vars_vals[i]
        if save_dir:
            np.save(save_dir, save_dict)
        return save_dict

    def restore_vars(self, sess, load_dir, map_fun):
        # TODO (Mehrdad --> Mehrdad) double check no redundant variables are transferred
        print('Trying to restore checkpoint')
        if isinstance(load_dir, str):
            vars_list = np.load(load_dir, allow_pickle=True).item()
        elif isinstance(load_dir, dict):
            vars_list = load_dir
        else:
            exit(1)
        # graph = tf.get_default_graph()
        # vars_to_restore = [graph.get_tensor_by_name(map_fun(var_name)) for var_name in vars_list if not map_fun(var_name)==None]
        restore_ops = [self.load_ops[var_name] for var_name in vars_list if not map_fun(var_name) is None]
        feed_dict = {self.vars_pl[var_name]: vars_list[var_name] for var_name in vars_list if
                     not map_fun(var_name) is None}
        # Make sure all the variables are found
        # assert  len(vars_to_restore) == len(vars_list)
        # sess.run([tf.assign(v, vars_list[v.name]) for v in vars_to_restore])
        sess.run(restore_ops, feed_dict=feed_dict)
        print('Restored successfully')
        return


def colormap(name='cityscapes'):
    if name == 'cityscapes':
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [128, 64, 128]
        colormap[1] = [244, 35, 232]
        colormap[2] = [70, 70, 70]
        colormap[3] = [102, 102, 156]
        colormap[4] = [190, 153, 153]
        colormap[5] = [153, 153, 153]
        colormap[6] = [250, 170, 30]
        colormap[7] = [220, 220, 0]
        colormap[8] = [107, 142, 35]
        colormap[9] = [152, 251, 152]
        colormap[10] = [70, 130, 180]
        colormap[11] = [220, 20, 60]
        colormap[12] = [255, 0, 0]
        colormap[13] = [0, 0, 142]
        colormap[14] = [0, 0, 70]
        colormap[15] = [0, 60, 100]
        colormap[16] = [0, 80, 100]
        colormap[17] = [0, 0, 230]
        colormap[18] = [119, 11, 32]
    else:
        raise Exception('Unknown colormap')

    return colormap


def calculate_miou(conf_matrix, population=False, detailed=False, nan=False):
    """
    Calculates MIOU based on confusion matrix

    :param conf_matrix: confusion matrix
    :type conf_matrix: list or np.ndarray
    :param population:
    :type population: bool
    :param detailed:
    :type detailed: bool
    :return: list of per class IoUs
    :rtype: list
    """
    miou = []
    false_pos = []
    false_neg = []
    number_of_classes = len(conf_matrix[0])
    for i in range(number_of_classes):
        denominator = 0
        for j in range(number_of_classes):
            denominator += conf_matrix[i][j]
            denominator += conf_matrix[j][i]
        denominator -= conf_matrix[i][i]
        if denominator == 0:
            if nan:
                miou.append(np.nan)
            else:
                miou.append('Not predicted/present')
            if detailed:
                false_pos.append(0)
                false_neg.append(0)
        else:
            miou.append(conf_matrix[i][i] / (max(denominator, 1)))
            if detailed:
                false_neg.append((np.sum(conf_matrix[i]) - conf_matrix[i][i]) / denominator)
                false_pos.append((np.sum(conf_matrix[:, i]) - conf_matrix[i][i]) / denominator)
    if population:
        population_class = np.sum(conf_matrix, axis=1)
        if detailed:
            return miou, population_class / np.sum(population_class), false_neg, false_pos
        else:
            return miou, population_class / np.sum(population_class)
    else:
        if detailed:
            return miou, false_neg, false_pos
        else:
            return miou


def mini_batch(deque_images, deque_labels, crop_size, scale, mini_batch_size, num_of_iterations, flip=False):
    """
    :type deque_images: deque or list
    :type deque_labels: deque or list
    :type crop_size: list
    :type scale: list or np.ndarray
    :type mini_batch_size: int
    :type num_of_iterations: int
    :type flip: bool
    """
    dict_scaled_images = {scale_choice: {} for scale_choice in scale}
    dict_scaled_labels = {scale_choice: {} for scale_choice in scale}
    output_batches_images = np.empty(
        (num_of_iterations, mini_batch_size, crop_size[0], crop_size[1], deque_images[0].shape[2]))
    output_batches_labels = np.empty((num_of_iterations, mini_batch_size, crop_size[0], crop_size[1]))
    deque_list_images = list(deque_images) if isinstance(deque_images, deque) else deque_images
    deque_list_labels = list(deque_labels) if isinstance(deque_labels, deque) else deque_labels
    total_size = len(deque_list_images)
    for i in range(num_of_iterations):
        for j in range(mini_batch_size):
            pic_index = np.random.choice(total_size)
            height_image = deque_list_images[pic_index].shape[0]
            width_image = deque_list_images[pic_index].shape[1]
            chosen_scale = scale[random.randint(0, len(scale)-1)]
            actual_scale = chosen_scale * crop_size[1] / width_image
            max_h = int(height_image * actual_scale) - crop_size[0]
            max_w = int(width_image * actual_scale) - crop_size[1]
            assert max_w >= 0
            assert max_h >= 0
            h = random.randint(0, max_h)
            w = random.randint(0, max_w)
            if pic_index not in dict_scaled_images[chosen_scale]:
                if actual_scale == 1 and chosen_scale == 1:
                    dict_scaled_images[chosen_scale][pic_index] = deque_list_images[pic_index]
                    dict_scaled_labels[chosen_scale][pic_index] = deque_list_labels[pic_index]
                else:
                    dict_scaled_images[chosen_scale][pic_index] = cv2.resize(deque_list_images[pic_index],
                                                                             (int(width_image * actual_scale),
                                                                              int(height_image * actual_scale)),
                                                                             interpolation=cv2.INTER_LINEAR)
                    dict_scaled_labels[chosen_scale][pic_index] = cv2.resize(deque_list_labels[pic_index],
                                                                             (int(width_image * actual_scale),
                                                                              int(height_image * actual_scale)),
                                                                             fx=0, fy=0,
                                                                             interpolation=cv2.INTER_NEAREST)
            w_end = w + crop_size[1]
            h_end = h + crop_size[0]
            if flip and np.random.random() > 0.5:
                output_batches_images[i][j] = np.flip(dict_scaled_images[chosen_scale][pic_index][h:h_end, w:w_end, :],
                                                      axis=1)
                output_batches_labels[i][j] = np.flip(dict_scaled_labels[chosen_scale][pic_index][h:h_end, w:w_end],
                                                      axis=1)
            else:
                output_batches_images[i][j] = dict_scaled_images[chosen_scale][pic_index][h:h_end, w:w_end, :]
                output_batches_labels[i][j] = dict_scaled_labels[chosen_scale][pic_index][h:h_end, w:w_end]

    return output_batches_images, output_batches_labels


def string_class_iou(class_iou_list, population=None, headers=None, class_weights=None):
    str_out = ""
    if headers is not None:
        str_out = "%22s\t" % ""
        for header in headers:
            str_out = str_out + header + "\t\t"
        str_out = str_out + "\n"
    labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
              'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    if class_weights is not None:
        labels = [labels[i] for i in np.where(class_weights == 1)[0]]
    if not type(class_iou_list[0]) == list:
        class_iou_list = [class_iou_list]
    for i in range(len(class_iou_list[0])):
        if population is not None:
            str_out = str_out + ("%-22s" % (labels[i] + "(%.3g):" % (population[i] * 100.0)))
        else:
            str_out = str_out + ("%-22s" % (labels[i] + ":"))
        str_out = str_out + "\t"
        for j in range(len(class_iou_list)):
            if type(class_iou_list[j][i]) == str:
                str_out = str_out + class_iou_list[j][i] + "\t"
            else:
                str_out = str_out + "%.1f" % (class_iou_list[j][i] * 100.0) + "\t\t\t"
        str_out = str_out + "\n"
    return str_out


def inspect(nodes):
    src = None
    dst = None
    for i, node in enumerate(nodes):
        if 'logits/semantic/Conv2D' in node.name:
            dst = i
        elif 'concat_projection/Relu' in node.name:
            src = i
    return src, dst


def prune(nodes, tag):
    new_nodes = []
    for n in nodes:
        if not tag in n.name:
            new_nodes.append(n)
        else:
            print('Removed:', n.name)
    return new_nodes


def choose_frames(frame_label_list, sample_fraction):
    """
    Choose equally-distanced frames based on the number of samples wanted

    :param frame_label_list: A list containing frame-label tuples
    :type frame_label_list: list of (np.ndarray, np.ndarray)
    :param sample_fraction: The fraction of samples wanted
    :type sample_fraction: float
    :return: list of sample indices
    :rtype: np.ndarray
    """
    samples = int(np.round(sample_fraction * len(frame_label_list)))
    indices = np.linspace(-1, len(frame_label_list)-1, samples+1, endpoint=True)[1:]
    indices = np.round(indices).astype(int)
    assert indices.size == samples, f"indices had {indices.size} values but samples is {samples}"
    frames_chosen = [frame_label_list[chosen_index][0] for chosen_index in indices]
    labels_chosen = [frame_label_list[chosen_index][1] for chosen_index in indices]
    return frames_chosen, labels_chosen
