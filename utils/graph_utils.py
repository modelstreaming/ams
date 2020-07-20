#!/usr/bin/python
from contextlib import ExitStack
import tensorflow as tf
import numpy as np
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
import sys
import copy

sys.path.append('../../.')
from ams.utils.utils import colormap, prune, inspect

# TODO: simplify code, remove the sys.append, test running it, merge with other profilers

NUM_CLASSES = 19
print('There are %d classes' % NUM_CLASSES)


def trim_graph(sess, nodes, output_name_list):
    # this is not robust
    # Fix the batch-norm is_training issue
    for n in nodes:
        if 'BatchNorm' in n.name and len(n.input) == 5 and not n.name.startswith(
                'gradients'):
            assert n.input[3].endswith('Const')
            assert n.input[4].endswith('Const_1')
            n.input[3] = n.input[3].rstrip('Const') + 'moving_mean/read'
            n.input[4] = n.input[4].rstrip('Const_1') + 'moving_variance/read'
    output_graph = tf.GraphDef()
    output_graph.node.extend(nodes)

    # Prune unnecessary placeholders & grad ops
    output_graph = tf.graph_util.convert_variables_to_constants(sess, output_graph, output_name_list)
    nodes = output_graph.node
    # Remove the dropout nodes
    src, dst = inspect(nodes)
    if src and dst:
        # print(src, dst, nodes[src].name, nodes[dst].name)
        nodes[dst].input[0] = nodes[src].name

    nodes = prune(nodes, 'dropout')
    print('bring me back')
    # for i, n in enumerate(nodes):
    #    if n.name == 'semantic':
    #        n.input[0] = 'logits/semantic/BiasAdd'
    # Rebuild a graph-def
    graph_def = tf.GraphDef()
    graph_def.node.extend(nodes)
    graph_def = tf.graph_util.extract_sub_graph(graph_def, dest_nodes=output_name_list)
    return graph_def

def convert_batchnorms(sess):
  nodes = sess.graph_def.node
  nodes = copy.deepcopy(sess.graph_def.node)
   
  for n in nodes:
    # Bind the BN patch ops variables to the original BN
    for i in range(len(n.input)):
      n.input[i] = n.input[i].replace('/FusedBatchNormV3_patch', '')
  # Rename the old BN ops from X to X_toremove
  for n in nodes:
    if n.op == 'FusedBatchNormV3' and not 'patch' in n.name:
      n.name = n.name + '_toremove'
  # Rename the new BN ops from X_patch to X
  for n in nodes:
    if 'patch' in n.name:
      n.name = n.name.replace('/FusedBatchNormV3_patch', '')
    try:
      n.attr['_class'].list.s[0] = (n.attr['_class'].list.s[0].decode('ascii').replace('/FusedBatchNormV3_patch','')).encode('ascii')
    except:
      pass
  # Rebuild the new graph_def       
  output_graph = tf.GraphDef()
  output_graph.node.extend(nodes)
  
  return output_graph
      

def trim_graph_frozen(sess, graph_def, input_name_list, output_name_list, kill_norms=False):
    # this is not robust
    # Fix the batch-norm is_training issue
    # TODO: clott
    for n in graph_def.node:
        if 'drop' in n.name:
            print(n)
    if kill_norms:
        graph_def = convert_batchnorms(sess) 
        # Prune the unused nodes
        graph_def = tf.graph_util.extract_sub_graph(graph_def, dest_nodes=output_name_list)
#        for n in graph_def.node:
#            if 'BatchNorm' in n.name and len(n.input) == 5 and not n.name.startswith(
#                    'gradients') and 'Momentum' not in n.name:
#                n.attr['is_training'].b = 0
#                assert n.attr['is_training'].b == 0
#                assert n.input[3].endswith('Const')
#                assert n.input[4].endswith('Const_1')
#                n.input[3] = n.input[3].rstrip('Const') + 'moving_mean/read'
#                n.input[4] = n.input[4].rstrip('Const_1') + 'moving_variance/read'
    gdef = strip_unused_lib.strip_unused(input_graph_def=graph_def,
                                         input_node_names=input_name_list,
                                         output_node_names=output_name_list,
                                         placeholder_type_enum=dtypes.float32.as_datatype_enum)

    gdef = tf.graph_util.convert_variables_to_constants(sess, gdef, output_name_list)
    # output_graph = tf.GraphDef()
    # output_graph.node.extend(graph_def.node)
    #
    # # Prune unnecessary placeholders & grad ops
    # output_graph = tf.graph_util.convert_variables_to_constants(sess, output_graph, output_name_list)
    # nodes = output_graph.node
    # # Remove the dropout nodes
    # src, dst = inspect(nodes)
    # if src and dst:
    #     print(src, dst, nodes[src].name, nodes[dst].name)
    #     nodes[dst].input[0] = nodes[src].name
    #
    # nodes = prune(nodes, 'dropout')
    # print('bring me back')
    # # for i, n in enumerate(nodes):
    # #    if n.name == 'semantic':
    # #        n.input[0] = 'logits/semantic/BiasAdd'
    # # Rebuild a graph-def
    # graph_def = tf.GraphDef()
    # graph_def.node.extend(nodes)
    # graph_def = tf.graph_util.extract_sub_graph(graph_def, dest_nodes=output_name_list)
    return gdef


def create_teacher(meta_dir, class_weights=None, test_mode=False):
    # predictions is the full 19 class label, miou_student is calculated with class weights
    if class_weights is not None:
        class_weights = np.where(class_weights == 1)[0]
    teacher_graph = None
    if not test_mode:
        teacher_graph = tf.Graph()
    with ExitStack() as stack:
        if not test_mode:
            stack.enter_context(teacher_graph.as_default())
        print('Loading the teacher graph from a saved meta')
        if test_mode:
            teacher_saver = tf.train.import_meta_graph('%s.meta' % meta_dir, clear_devices=True, import_scope='teacher')
            teacher_images = tf.get_default_graph().get_tensor_by_name('teacher/images:0')
            teacher_predictions = tf.get_default_graph().get_tensor_by_name('teacher/predictions:0')
            teacher_logits = tf.get_default_graph().get_tensor_by_name('teacher/logits/semantic/BiasAdd:0')
            teacher_logits = tf.image.resize_bilinear(teacher_logits, tf.shape(teacher_images)[1:3])
            label_pl = tf.placeholder(tf.int32, [None, None, None], name='teacher/label')
        else:
            teacher_saver = tf.train.import_meta_graph('%s.meta' % meta_dir)
            teacher_images = tf.get_default_graph().get_tensor_by_name('images:0')
            teacher_predictions = tf.get_default_graph().get_tensor_by_name('predictions:0')
            teacher_logits = tf.get_default_graph().get_tensor_by_name('logits/semantic/BiasAdd:0')
            teacher_logits = tf.image.resize_bilinear(teacher_logits, tf.shape(teacher_images)[1:3])
            label_pl = tf.placeholder(tf.int32, [None, None, None], name='label')
        shape_teacher = tf.shape(teacher_logits)
        teacher_logits_resized = tf.slice(teacher_logits, [0, 1, 1, 0],
                                  [shape_teacher[0], shape_teacher[1] - 1, shape_teacher[2] - 1, shape_teacher[3]])
        teacher_logits_resized = tf.image.resize_nearest_neighbor(teacher_logits_resized, (256, 512), align_corners=True)
        teacher_probs = tf.nn.softmax(teacher_logits, axis=-1)
        teacher_probs = tf.reduce_max(teacher_probs, axis=-1)
        teacher_predictions_reduced = None
        teacher_predictions_one_hot_reduced = None
        weights = None
        if class_weights is not None:
            # TODO: maybe we can gather logits and then argmax to save one step of one hot
            teacher_predictions_one_hot = tf.one_hot(teacher_predictions, NUM_CLASSES, axis=-1)
            teacher_predictions_one_hot_reduced = tf.gather(teacher_predictions_one_hot, class_weights, axis=-1)
            teacher_predictions_reduced = tf.argmax(teacher_predictions_one_hot_reduced, axis=-1, output_type=tf.int32)
            weights = tf.reduce_max(teacher_predictions_one_hot_reduced, axis=-1)
            mean_iou_student, update_op_student = tf.metrics.mean_iou(
                labels=teacher_predictions_reduced,
                predictions=label_pl,
                num_classes=len(class_weights),
                weights=weights)
        else:
            mean_iou_student, update_op_student = tf.metrics.mean_iou(
                labels=teacher_predictions,
                predictions=label_pl,
                num_classes=NUM_CLASSES)
        mean_iou_teacher, update_op_teacher = tf.metrics.mean_iou(
            labels=label_pl,
            predictions=teacher_predictions,
            num_classes=NUM_CLASSES)
    teacher = {'graph': teacher_graph, 'images': teacher_images, 'predictions': teacher_predictions,
               'student_label_pl': label_pl, 'miou_hat': mean_iou_student, 'update_op_hat': update_op_student,
               'miou_teacher': mean_iou_teacher, 'update_op_teacher': update_op_teacher, 'logits': teacher_logits,
               'predictions_reduced': teacher_predictions_reduced, 'weights': weights, 'probabilities': teacher_probs,
               'predictions_one_hot_reduced': teacher_predictions_one_hot_reduced, 'logits_sml': teacher_logits_resized}
    return teacher


def create_teacher_v2(meta_dir, class_weights=False, test_mode=False):
    # predictions is the full 19 class label, miou_student is calculated with class weights
    teacher_graph = None
    if not test_mode:
        teacher_graph = tf.Graph()
    with ExitStack() as stack:
        if not test_mode:
            stack.enter_context(teacher_graph.as_default())
        print('Loading the teacher graph from a saved meta')
        if test_mode:
            teacher_saver = tf.train.import_meta_graph('%s.meta' % meta_dir, clear_devices=True, import_scope='teacher')
            teacher_images = tf.get_default_graph().get_tensor_by_name('teacher/images:0')
            teacher_predictions = tf.get_default_graph().get_tensor_by_name('teacher/predictions:0')
            teacher_logits = tf.get_default_graph().get_tensor_by_name('teacher/logits/semantic/BiasAdd:0')
            teacher_logits = tf.image.resize_bilinear(teacher_logits, tf.shape(teacher_images)[1:3])
            label_pl = tf.placeholder(tf.int32, [None, None, None], name='teacher/label')
        else:
            teacher_saver = tf.train.import_meta_graph('%s.meta' % meta_dir)
            teacher_images = tf.get_default_graph().get_tensor_by_name('images:0')
            teacher_predictions = tf.get_default_graph().get_tensor_by_name('predictions:0')
            teacher_logits = tf.get_default_graph().get_tensor_by_name('logits/semantic/BiasAdd:0')
            teacher_logits = tf.image.resize_bilinear(teacher_logits, tf.shape(teacher_images)[1:3])
            label_pl = tf.placeholder(tf.int32, [None, None, None], name='label')
        shape_teacher = tf.shape(teacher_logits)
        teacher_predictions = tf.slice(teacher_predictions, [0, 1, 1],
                                       [shape_teacher[0], shape_teacher[1] - 1, shape_teacher[2] - 1])
        teacher_predictions_small = tf.expand_dims(teacher_predictions, axis=-1)
        teacher_predictions_small = tf.image.resize_nearest_neighbor(teacher_predictions_small, (256, 512))
        teacher_predictions_small = tf.squeeze(teacher_predictions_small, axis=-1)
        # label_pl_large = tf.expand_dims(label_pl, axis=-1)
        # label_pl_large = tf.image.resize_nearest_neighbor(label_pl_large, (1024, 2048))
        # label_pl_large = tf.squeeze(label_pl_large, axis=-1)
        teacher_logits = tf.slice(teacher_logits, [0, 1, 1, 0],
                                  [shape_teacher[0], shape_teacher[1] - 1, shape_teacher[2] - 1, shape_teacher[3]])
        teacher_probs = tf.nn.softmax(teacher_logits, axis=-1)
        teacher_probs = tf.reduce_max(teacher_probs, axis=-1)
        teacher_predictions_reduced = None
        teacher_predictions_one_hot_reduced = None
        weights = None
        class_weights_pl = None
        if class_weights:
            class_weights_pl = tf.placeholder(tf.int32, [None])
            # TODO: maybe we can gather logits and then argmax to save one step of one hot
            teacher_predictions_one_hot = tf.one_hot(teacher_predictions_small, NUM_CLASSES, axis=-1)
            # teacher_predictions_one_hot = tf.one_hot(teacher_predictions, NUM_CLASSES, axis=-1)
            teacher_predictions_one_hot_reduced = tf.gather(teacher_predictions_one_hot, class_weights_pl, axis=-1)
            teacher_predictions_reduced = tf.argmax(teacher_predictions_one_hot_reduced, axis=-1, output_type=tf.int32)
            weights = tf.reduce_max(teacher_predictions_one_hot_reduced, axis=-1)
            mean_iou_student, update_op_student = tf.metrics.mean_iou(
                labels=teacher_predictions_reduced,
                # predictions=label_pl_large,
                predictions=label_pl,
                num_classes=NUM_CLASSES,
                weights=weights)
            curr_num_of_classes = tf.shape(teacher_predictions_one_hot_reduced)[3]
            update_op_student = tf.slice(update_op_student, [0, 0], [curr_num_of_classes, curr_num_of_classes])
        else:
            mean_iou_student, update_op_student = tf.metrics.mean_iou(
                labels=teacher_predictions,
                predictions=label_pl,
                num_classes=NUM_CLASSES)
        mean_iou_teacher, update_op_teacher = tf.metrics.mean_iou(
            labels=label_pl,
            predictions=teacher_predictions,
            num_classes=NUM_CLASSES)
    teacher = {'graph': teacher_graph, 'images': teacher_images, 'predictions': teacher_predictions,
               'student_label_pl': label_pl, 'miou_hat': mean_iou_student, 'update_op_hat': update_op_student,
               'miou_teacher': mean_iou_teacher, 'update_op_teacher': update_op_teacher, 'logits': teacher_logits,
               'predictions_reduced': teacher_predictions_reduced, 'weights': weights, 'probabilities': teacher_probs,
               'predictions_one_hot_reduced': teacher_predictions_one_hot_reduced, 'class indices': class_weights_pl,
               'predictions_sml': teacher_predictions_small}
    return teacher


def prob_confmat(student_labels, teacher_probs, num_classes):
    mat = []
    student_labels = tf.reshape(student_labels, [-1])
    teacher_probs = tf.reshape(teacher_probs, [-1, tf.shape(teacher_probs)[-1]])
    for i in range(num_classes):
        mat.append(tf.expand_dims(tf.where(tf.equal(student_labels,i), teacher_probs, tf.zeros_like(teacher_probs)), axis=-1))
    mat = tf.concat(mat, axis=-1)
    mat = tf.reduce_sum(mat, axis=0)
    mat.set_shape([num_classes, num_classes])

    acc_mat = tf.Variable(tf.zeros_like(mat), trainable=False, collections=[], name='prob_conf_mat')
    reset = tf.initializers.variables([acc_mat], name='init_prob_confmat')
    update = tf.assign(acc_mat, acc_mat + mat)

    row_sum = tf.reduce_sum(acc_mat, axis=1)
    col_sum = tf.reduce_sum(acc_mat, axis=0)
    tp = tf.diag_part(acc_mat)
    iou = tp / (row_sum + col_sum - tp)
    miou = tf.reduce_mean(iou)

    return update, miou, reset


def prob_confmat_star(student_labels, teacher_labels, weights, teacher_probs, num_classes):
    mat_stu = []
    student_labels = tf.reshape(student_labels, [-1])
    weights = tf.reshape(weights, [-1])
    teacher_probs = tf.reshape(teacher_probs, [-1, tf.shape(teacher_probs)[-1]])
    for i in range(num_classes):
        mat_stu.append(tf.expand_dims(tf.where(tf.logical_and(tf.equal(student_labels, i), tf.not_equal(weights, 0)),
                                               teacher_probs, tf.zeros_like(teacher_probs)), axis=-1))
    mat_stu = tf.concat(mat_stu, axis=-1)
    mat_stu = tf.reduce_sum(mat_stu, axis=0)
    mat_stu.set_shape([num_classes, num_classes])

    mat_star = []
    teacher_labels = tf.reshape(teacher_labels, [-1])
    teacher_probs = tf.reshape(teacher_probs, [-1, tf.shape(teacher_probs)[-1]])
    for i in range(num_classes):
        mat_star.append(
            tf.expand_dims(tf.where(tf.logical_and(tf.equal(teacher_labels, i), tf.not_equal(weights, 0)),
                                    teacher_probs, tf.zeros_like(teacher_probs)), axis=-1))
    mat_star = tf.concat(mat_star, axis=-1)
    mat_star = tf.reduce_sum(mat_star, axis=0)
    mat_star.set_shape([num_classes, num_classes])

    acc_mat_stu = tf.Variable(tf.zeros_like(mat_stu), trainable=False, collections=[], name='prob_conf_mat_stu')
    acc_mat_star = tf.Variable(tf.zeros_like(mat_star), trainable=False, collections=[], name='prob_conf_mat_star')
    reset = tf.initializers.variables([acc_mat_stu, acc_mat_star], name='init_prob_confmat')
    update_stu = tf.assign(acc_mat_stu, acc_mat_stu + mat_stu)
    update_star = tf.assign(acc_mat_star, acc_mat_star + mat_star)

    return update_stu, update_star, reset


def prob_confmat_test():
    num_classes = 7
    student_labels = tf.random_uniform(minval=0, maxval=num_classes, shape=[10, 20, 30], dtype=tf.int32)
    teacher_probs = tf.random_uniform(minval=0, maxval=1, shape=[10, 20, 30, num_classes], dtype=tf.float32)
    teacher_probs = teacher_probs / tf.reduce_sum(teacher_probs, axis=-1, keepdims=True)
    update, miou, reset = prob_confmat(student_labels, teacher_probs, num_classes)
    with tf.Session() as sess:
        sess.run(reset)
        for i in range(3):
            print(sess.run([miou,update]))
            print(sess.run(miou))

def create_student_v3_test():
    meta_dir = '/data4/ModelStreaming/clean/models_inventory/mnv2_decay0.9_drop0.1/model'
    class_weights = np.array([1]*19)
    create_student_v3(meta_dir, class_weights=class_weights, threshold=None, map_misc=0, test_mode=False,
                      train_biases_only=False, regularize=False, soft_teacher=False, masked_gradients=True)

def create_student_v3(meta_dir, class_weights=None, threshold=None, map_misc=0, test_mode=False,
                      train_biases_only=False, regularize=False, soft_teacher=False, masked_gradients=False):
    if class_weights is not None:
        class_weights = np.where(class_weights == 1)[0]
    student_graph = None
    if not test_mode:
        student_graph = tf.Graph()
    with student_graph.as_default():#ExitStack() as stack:
        #if not test_mode:
        #    stack.enter_context(student_graph.as_default())
        #print('Loading the student graph from a saved meta')
        str_prepend = ""
        saver = tf.train.import_meta_graph('%s.meta' % meta_dir)
        if 'student_logits' in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            str_prepend = "student_"
        features_input = tf.get_default_graph().get_tensor_by_name('features_input:0')
        labels_input = tf.get_default_graph().get_tensor_by_name('labels_input:0')
        student_logits = tf.get_default_graph().get_tensor_by_name(str_prepend + 'logits:0')
        fill_input_buffer = tf.get_default_graph().get_operation_by_name('fill_input_buffer')
        features = tf.get_default_graph().get_tensor_by_name('features:0')
        labels = tf.get_default_graph().get_tensor_by_name('labels:0')
        teacher_labels_logits_pl = tf.placeholder(shape=[None, None, None, None], dtype=tf.float32)
        student_probs = tf.nn.softmax(student_logits, axis=-1)
        student_probs = tf.reduce_max(student_probs, axis=-1)
        # Create new BN ops
        nodes = student_graph.as_graph_def().node
        for n in nodes:
          if n.op == 'FusedBatchNormV3' and not 'patch' in n.name:
            assert len(n.input) == 5
            input_name = n.input[0]
            input_tensor = tf.get_default_graph().get_tensor_by_name('%s:0'%input_name)
            new_out = tf.layers.batch_normalization(input_tensor, fused=True, training=False, trainable=True, name='%s_patch'%n.name)

        learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
        if class_weights is not None:
            filtered_logits = tf.gather(student_logits, class_weights, axis=-1)
            filtered_logits = tf.identity(filtered_logits, "logits_reduced")
            filtered_teacher_labels_logits = tf.gather(teacher_labels_logits_pl, class_weights, axis=-1) 
            filtered_teacher_labels_probs = tf.nn.softmax(filtered_teacher_labels_logits, axis=-1)


            ################
#            for i in range(100):
#                print('adding an extra layer')
#            out_degree = filtered_logits.shape[-1]
#            filtered_logits = tf.concat([tf.stop_gradient(filtered_logits), features], axis=-1)
#            filtered_logits = tf.compat.v1.layers.conv2d(tf.stop_gradient(filtered_logits), 256, [3,3], padding='same', name='extra1', activation=tf.nn.relu)
#            filtered_logits = tf.compat.v1.layers.conv2d(filtered_logits, 128, [3,3], padding='same', name='extra2', activation=tf.nn.relu)
#            filtered_logits = tf.compat.v1.layers.conv2d(filtered_logits, out_degree, [3,3], padding='same', name='extra3')
            ######
            filtered_probs = tf.nn.softmax(filtered_logits, axis=-1)
            filtered_probs = tf.reduce_max(filtered_probs, axis=-1)
            filtered_predictions = tf.argmax(filtered_logits, axis=-1, output_type=tf.int32)
            filtered_predictions = tf.identity(filtered_predictions, str_prepend + 'predictions')
            labels_onehot = tf.one_hot(tf.cast(labels, tf.int32), NUM_CLASSES, axis=-1)
            filtered_labels_onehot = tf.gather(labels_onehot, class_weights, axis=-1)
            filtered_labels = tf.argmax(filtered_labels_onehot, axis=-1)
            # soft_update, soft_miou, soft_reset = prob_confmat(filtered_labels, filtered_teacher_labels_probs, len(class_weights))
            # set weight to zero if all elements in the last dimension are zero
            weights = tf.reduce_sum(filtered_labels_onehot, axis=-1)
            mean_iou, update_op = tf.metrics.mean_iou(
                labels=tf.reshape(filtered_labels, [-1, 1]),
                predictions=tf.reshape(filtered_predictions, [-1, 1]),
                num_classes=len(class_weights),
                weights=tf.reshape(weights, [-1, 1]))
            if soft_teacher:
                pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_teacher_labels_probs)
            else:
                pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_labels_onehot)
            weights = tf.cast(weights, tf.bool)
            loss = tf.reduce_mean(tf.boolean_mask(pixel_loss, weights))
            loss_sum = None
            for i in range(len(class_weights)):
                weights_choice = tf.equal(filtered_labels, i)
                weights_choice = tf.logical_or(weights_choice, tf.equal(filtered_predictions, tf.cast(i, tf.int32)))
                loss_selective = tf.reduce_mean(tf.boolean_mask(pixel_loss, tf.logical_and(weights, weights_choice)))
                if loss_sum is None:
                    loss_sum = loss_selective
                else:
                    loss_sum = loss_sum + loss_selective
            loss_selective = tf.reduce_mean(loss_sum)

        tvars = tf.trainable_variables()
        entire_model_vars = [var for var in tvars if not 'image_cache' in var.name and not 'patch' in var.name]
        tvars = [var for var in tvars if not 'image_cache' in var.name and not 'patch' in var.name]
        tvars = [var for var in tvars if not any([(x in var.name) for x in ['conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12']])]
        #tvars = [var for var in tvars if any([(x in var.name) for x in ['semantic', 'concat_projection', 'aspp0', 'image_pooling', 'expanded_conv_16']])]
        def var_size(v):
            size = 1
            for s in v.shape:
                size = size * s
            return size
        tvars = [var for var in tvars if var_size(var) < 1e4]
        #tvars = [var for var in tvars if var_size(var) < 320*960] 
        #tvars = [var for var in tvars if not 'BatchNorm' in var.name]
        #tvars = [var for var in tvars if not 'weights' in var.name]
        #tvars = [var for var in tvars if 'weights' in var.name]
        #tvars = [var for var in tvars if 'extra' in var.name]
        chk = np.load('%s.npy'%meta_dir, allow_pickle=True).item()
        drift_loss = 0.
        for v in tvars:
            drift_loss = drift_loss + tf.reduce_sum(tf.square(chk[v.name] - v))

        #loss = loss + 0.01 * drift_loss
        #print('Training:', tvars)
        total_size = 0
        for v in tvars:
            size = 1
            for s in v.shape:
                size = size * s
            total_size = total_size + size
        #print('Total Number of Training Vars: %d'%total_size) 

        if train_biases_only:
            tvars = [var for var in tvars if not 'weight' in var.name]
        if regularize:
            for i in range(10):
                print('regularizing')
            loss = loss + 0.01 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tvars])
        update_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #print('update_ops', update_ops)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # with tf.control_dependencies(update_bn):
        if True:
            #optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
            #train = optimizer.minimize(loss, var_list=tvars)
            # grads_and_vars = optimizer.compute_gradients(loss, var_list=entire_model_vars)
            # snr = {v.name: tf.norm(g) for (g, v) in grads_and_vars}

            grad_masks_pl = None
            train = None
            modified_train = None
            # if masked_gradients:
            #     grad_masks_pl = {v.name: tf.placeholder(shape=v.shape, dtype=tf.bool) for (g,v) in grads_and_vars}
            #     grads_and_vars = [(tf.where(grad_masks_pl[v.name], g, tf.zeros_like(g)), v) for (g,v) in grads_and_vars]

            # train_entire_model = optimizer.apply_gradients(grads_and_vars)
            # train_entire_model = optimizer.minimize(loss, var_list=entire_model_vars)
            # train = train_entire_model
            #first_moment = {v.name: optimizer.get_slot(v, 'm') for v in entire_model_vars}

            # train_update_ops = train_entire_model.control_inputs
            # weight_train_ops = {k.name: [o for o in train_update_ops if k.name.rstrip(':0') in o.name] for k in entire_model_vars}

            if masked_gradients:
                grad_masks_pl = {v.name: tf.placeholder(shape=v.shape, dtype=tf.bool) for v in entire_model_vars}
                backup_vars = {v.name: tf.Variable(tf.zeros_like(v), name='%s_copy' % v.name.rstrip(':0'),
                                                   trainable=False) for v in entire_model_vars}
                main_vars = {v.name: v for v in entire_model_vars}
                with tf.control_dependencies(update_bn):
                    backup_ops = [tf.assign(backup_vars[k], main_vars[k], use_locking=True) for k in backup_vars]
                with tf.control_dependencies(backup_ops):
                    train_all = optimizer.minimize(loss)
                with tf.control_dependencies([train_all]):
                    modified_train = [tf.assign(main_vars[k], tf.where(grad_masks_pl[k], main_vars[k], backup_vars[k]),
                                                use_locking=True) for k in main_vars]
            else:
                with tf.control_dependencies(update_bn):
                    train = optimizer.minimize(loss)
            # train_selective = tf.train.AdamOptimizer(learning_rate).minimize(loss_selective, var_list=tvars)
        student_saver = tf.train.Saver()
    student = {'graph': student_graph,
               'features_input': features_input,
               'labels_input': labels_input,
               'fill_input_buffer': fill_input_buffer,
               'predictions': filtered_predictions,
               'train': train,
               'train_coord': modified_train,
               # 'train_sel': train_selective,
               'saver': saver,
               'logits': student_logits,
               'miou': mean_iou,
               'update_op': update_op,
               'learning_rate': learning_rate,
               'logits_reduced': filtered_logits,
               'weights': weights,
               'loss': loss,
               'loss_sel': loss_selective,
               'labels_reduced': filtered_labels,
               'features': features,
               'probabilities': student_probs,
               'probabilities_reduced': filtered_probs,
               "prepend": str_prepend,
               'teacher_labels_logits_pl': teacher_labels_logits_pl,
               'training_var_names': [v.name for v in tvars],
               # 'train_entire_model': train_entire_model,
               'update_bn': update_bn,
               # 'weight_train_ops': weight_train_ops,
               # 'snr': snr,
               'grad_masks_pl': grad_masks_pl,
               #'first_moment': first_moment,
               # 'soft_update_op': soft_update,
               # 'soft_miou': soft_miou,
               # 'soft_reset': soft_reset,
               }
    return student

def create_student_v2(meta_dir, class_weights=None, threshold=None, map_misc=0, test_mode=False,
                      train_biases_only=False, regularize=False):
    if class_weights is not None:
        class_weights = np.where(class_weights == 1)[0]
    student_graph = None
    if not test_mode:
        student_graph = tf.Graph()
    with ExitStack() as stack:
        if not test_mode:
            stack.enter_context(student_graph.as_default())
        print('Loading the student graph from a saved meta')
        str_prepend = ""
        saver = tf.train.import_meta_graph('%s.meta' % meta_dir)
        if 'student_logits' in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            str_prepend = "student_"
        features_input = tf.get_default_graph().get_tensor_by_name('features_input:0')
        labels_input = tf.get_default_graph().get_tensor_by_name('labels_input:0')
        student_logits = tf.get_default_graph().get_tensor_by_name(str_prepend + 'logits:0')
        fill_input_buffer = tf.get_default_graph().get_operation_by_name('fill_input_buffer')

        # The nodes below mainly stand for the inputs of frozen/tflite graphs
        # Note: 'StopGradient:0' can be changed to 'features:0' and is kept here for backward compability
        student_StopGradient = tf.get_default_graph().get_tensor_by_name('StopGradient:0')
        features = tf.get_default_graph().get_tensor_by_name('features:0')
        labels = tf.get_default_graph().get_tensor_by_name('labels:0')

        if class_weights is not None:
            student_reduced_logits = tf.gather(student_logits, class_weights, axis=-1)
            if threshold is not None:
                student_reduced_probs = tf.nn.softmax(student_reduced_logits)
                initial_student_predictions = tf.argmax(student_reduced_probs, axis=-1, output_type=tf.int32)
                student_predictions = tf.where(tf.reduce_max(student_reduced_probs, axis=-1) > threshold,
                                               initial_student_predictions,
                                               tf.ones_like(initial_student_predictions) * map_misc)
            else:
                student_predictions = tf.argmax(student_reduced_logits, axis=-1, output_type=tf.int32)

        else:
            initial_student_predictions = tf.argmax(student_logits, axis=-1, output_type=tf.int32)
            if threshold is not None:
                student_predictions = tf.where(tf.reduce_max(student_logits, axis=-1) > threshold,
                                               initial_student_predictions,
                                               tf.ones_like(initial_student_predictions) * map_misc)
            else:
                student_predictions = initial_student_predictions
        student_predictions = tf.identity(student_predictions, str_prepend + 'predictions')
        learning_rate = tf.placeholder(dtype=tf.float64, shape=(), name='learning_rate')
        logits_reduced = None
        labels_reduced = None
        weights = None
        if class_weights is not None:
            labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), NUM_CLASSES, axis=-1)
            labels_one_hot_reduced = tf.gather(labels_one_hot, class_weights, axis=-1)
            labels_reduced = tf.argmax(labels_one_hot_reduced, axis=-1)
            logits_reduced = tf.gather(student_logits, class_weights, axis=-1)
            # TODO: I changed this
            logits_reduced = tf.identity(logits_reduced, "logits_reduced")
            weights = tf.reduce_max(labels_one_hot_reduced, axis=-1)
            mean_iou, update_op = tf.metrics.mean_iou(
                labels=tf.reshape(labels_reduced, [-1, 1]),
                predictions=tf.reshape(student_predictions, [-1, 1]),
                num_classes=len(class_weights),
                weights=tf.reshape(weights, [-1, 1]))
            pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_reduced, labels=labels_one_hot_reduced)

            weights = tf.cast(weights, tf.bool)
            loss = tf.reduce_mean(tf.boolean_mask(pixel_loss, weights))

        else:
            mean_iou, update_op = tf.metrics.mean_iou(
                labels=tf.reshape(labels, [-1, 1]),
                predictions=tf.reshape(student_predictions, [-1, 1]),
                num_classes=NUM_CLASSES)
            loss = tf.reduce_mean(tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits(logits=student_logits,
                                                                                          labels=tf.one_hot(
                                                                                              tf.stop_gradient(labels),
                                                                                              NUM_CLASSES)),
                                                  labels < NUM_CLASSES))
        tvars = tf.trainable_variables()
        tvars = [var for var in tvars if not 'image_cache' in var.name]
        if train_biases_only:
            tvars = [var for var in tvars if not 'weight' in var.name]
            print(tvars)
        if regularize:
            loss = loss + 0.0001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tvars])
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tvars)

        student_saver = tf.train.Saver()
    student = {'graph': student_graph,
               'features_input': features_input,
               'labels_input': labels_input,
               'fill_input_buffer': fill_input_buffer,
               'predictions': student_predictions,
               'train': train,
               'saver': saver,
               'StopGradient': student_StopGradient,
               'logits': student_logits,
               'miou': mean_iou,
               'update_op': update_op,
               'learning_rate': learning_rate,
               'logits_reduced': logits_reduced,
               'weights': weights,
               'loss': loss,
               'labels_reduced': labels_reduced,
               "prepend": str_prepend}
    return student


def create_student(meta_dir, class_weights=None, threshold=None, map_misc=0, test_mode=False, train_biases_only=False):
    if class_weights is not None:
        class_weights = np.where(class_weights == 1)[0]
    student_graph = None
    if not test_mode:
        student_graph = tf.Graph()
    with ExitStack() as stack:
        if not test_mode:
            stack.enter_context(student_graph.as_default())
        print('Loading the student graph from a saved meta')
        str_prepend = ""
        saver = tf.train.import_meta_graph('%s.meta' % meta_dir)
        if 'student_logits' in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            str_prepend = "student_"
        image_pl = tf.get_default_graph().get_tensor_by_name('image:0')
        is_inference = tf.get_default_graph().get_tensor_by_name('is_inference:0')
        load_image = tf.get_default_graph().get_tensor_by_name('load_image:0')
        student_logits = tf.get_default_graph().get_tensor_by_name(str_prepend + 'logits:0')
        student_StopGradient = tf.get_default_graph().get_tensor_by_name('StopGradient:0')
        if class_weights is not None:
            # tf_class_weights_mult = tf.constant(np.reshape(class_weights, (1, 1, 1, NUM_CLASSES)), dtype=tf.float32)
            # student_probs = tf.nn.softmax(student_logits)
            # student_reduced_probs = student_probs * tf_class_weights_mult
            # initial_student_predictions = tf.argmax(student_reduced_probs, axis=-1, output_type=tf.int32)
            # TODO: we can probably skip the softmax here
            # student_probs = tf.nn.softmax(student_logits)
            # student_reduced_probs = tf.gather(student_probs, class_weights, axis=-1)
            # initial_student_predictions = tf.argmax(student_reduced_probs, axis=-1, output_type=tf.int32)
            student_reduced_logits = tf.gather(student_logits, class_weights, axis=-1)
            if threshold is not None:
                student_reduced_probs = tf.nn.softmax(student_reduced_logits)
                initial_student_predictions = tf.argmax(student_reduced_probs, axis=-1, output_type=tf.int32)
                student_predictions = tf.where(tf.reduce_max(student_reduced_probs, axis=-1) > threshold,
                                               initial_student_predictions,
                                               tf.ones_like(initial_student_predictions) * map_misc)
            else:
                student_predictions = tf.argmax(student_reduced_logits, axis=-1, output_type=tf.int32)

        else:
            initial_student_predictions = tf.argmax(student_logits, axis=-1, output_type=tf.int32)
            if threshold is not None:
                student_predictions = tf.where(tf.reduce_max(student_logits, axis=-1) > threshold,
                                               initial_student_predictions,
                                               tf.ones_like(initial_student_predictions) * map_misc)
            else:
                student_predictions = initial_student_predictions
        student_predictions = tf.identity(student_predictions, str_prepend + 'predictions')
        label_pl = tf.placeholder(tf.int32, [None, None, None], name=(str_prepend + 'label'))
        label_cache = tf.Variable(np.zeros([1, 1024, 2048]), dtype=tf.int32, name='label_cache')
        load_label = tf.assign(label_cache, label_pl, validate_shape=False)
        load_label = tf.identity(load_label, name='load_label')
        labels = tf.cond(is_inference, lambda: label_pl, lambda: label_cache)
        labels = tf.stop_gradient(labels)
        learning_rate = tf.placeholder(dtype=tf.float64, shape=(), name='learning_rate')
        logits_reduced = None
        labels_reduced = None
        weights = None
        if class_weights is not None:
            # labels_one_hot = tf.one_hot(labels, NUM_CLASSES)
            # labels_one_hot_reduced = labels_one_hot * tf_class_weights_mult
            # weights = tf.reduce_max(labels_one_hot_reduced, axis=-1)
            # labels_reduced = tf.argmax(labels_one_hot_reduced, axis=-1, output_type=tf.int32)
            # logits_reduced = student_logits * tf_class_weights_mult
            # mean_iou, update_op = tf.metrics.mean_iou(
            #     labels=labels_reduced,
            #     predictions=student_predictions,
            #     num_classes=NUM_CLASSES)
            # pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_reduced, labels=labels_one_hot_reduced)
            # weights = tf.cast(weights, tf.bool)
            # weights = tf.math.logical_and(weights, labels < NUM_CLASSES)
            # loss = tf.reduce_mean(tf.boolean_mask(pixel_loss, weights))
            labels_one_hot = tf.one_hot(tf.stop_gradient(labels), NUM_CLASSES, axis=-1)
            labels_one_hot_reduced = tf.gather(labels_one_hot, class_weights, axis=-1)
            labels_reduced = tf.argmax(labels_one_hot_reduced, axis=-1)
            logits_reduced = tf.gather(student_logits, class_weights, axis=-1)
            weights = tf.reduce_max(labels_one_hot_reduced, axis=-1)
            mean_iou, update_op = tf.metrics.mean_iou(
                labels=labels_reduced,
                predictions=student_predictions,
                num_classes=len(class_weights),
                weights=weights)
            pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_reduced, labels=labels_one_hot_reduced)
            weights = tf.cast(weights, tf.bool)
            loss = tf.reduce_mean(tf.boolean_mask(pixel_loss, weights))
        else:
            mean_iou, update_op = tf.metrics.mean_iou(
                labels=labels,
                predictions=student_predictions,
                num_classes=NUM_CLASSES)
            loss = tf.reduce_mean(tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits(logits=student_logits,
                                                                                          labels=tf.one_hot(
                                                                                              tf.stop_gradient(labels),
                                                                                              NUM_CLASSES)),
                                                  labels < NUM_CLASSES))
        tvars = tf.trainable_variables()
        tvars = [var for var in tvars if not 'image_cache' in var.name]
        if train_biases_only:
            tvars = [var for var in tvars if not 'weight' in var.name]
            print(tvars)
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tvars)
        student_saver = tf.train.Saver()
    student = {'graph': student_graph, 'image_pl': image_pl, 'load_image': load_image,
               'predictions': student_predictions, 'train': train, 'load_label': load_label, 'label_pl': label_pl,
               'is_inference': is_inference, 'saver': saver, 'StopGradient': student_StopGradient,
               'logits': student_logits, 'miou': mean_iou, 'update_op': update_op, 'learning_rate': learning_rate,
               'logits_reduced': logits_reduced, 'weights': weights, 'loss': loss, 'labels_reduced': labels_reduced,
               "prepend": str_prepend}
    return student


def create_client(meta_dir_or_bytes, class_weights=None, threshold=None, map_misc=0):
    colormap_ = colormap()
    if class_weights is not None:
        class_weights = np.where(class_weights == 1)[0]
        colormap_ = np.take(colormap_, class_weights, axis=0)
    client_graph = tf.Graph()
    with client_graph.as_default():
        print('Loading the client graph from the meta')
        if isinstance(meta_dir_or_bytes, str):
            tf.train.import_meta_graph('%s.meta' % meta_dir_or_bytes)
        elif isinstance(meta_dir_or_bytes, bytes):
            tf.train.import_meta_graph(meta_dir_or_bytes)
        else:
            raise ValueError('Expected either file directory string or bytes for the meta')
        str_prepend = ''
        if 'student_logits' in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            str_prepend = 'student_'
        image_pl = tf.get_default_graph().get_tensor_by_name('image:0')
        is_inference = tf.get_default_graph().get_tensor_by_name('is_inference:0')
        load_image = tf.get_default_graph().get_tensor_by_name('load_image:0')
        logits = tf.get_default_graph().get_tensor_by_name(str_prepend + 'logits:0')
        if class_weights is not None:
            # tf_class_weights_mult = tf.constant(np.reshape(class_weights, (1, 1, 1, NUM_CLASSES)), dtype=tf.float32)
            # student_probs = tf.nn.softmax(student_logits)
            # student_reduced_probs = student_probs * tf_class_weights_mult
            # initial_student_predictions = tf.argmax(student_reduced_probs, axis=-1, output_type=tf.int32)
            if threshold is not None:
                student_probs = tf.nn.softmax(logits)
                student_reduced_probs = tf.gather(student_probs, class_weights, axis=-1)
                initial_student_predictions = tf.argmax(student_reduced_probs, axis=-1, output_type=tf.int32)
                predictions = tf.where(tf.reduce_max(student_reduced_probs, axis=-1) > threshold,
                                       initial_student_predictions,
                                       tf.ones_like(initial_student_predictions) * map_misc)
            else:
                student_reduced_logits = tf.gather(logits, class_weights, axis=-1)
                predictions = tf.argmax(student_reduced_logits, axis=-1, output_type=tf.int32)

        else:
            initial_student_predictions = tf.get_default_graph().get_tensor_by_name(str_prepend + 'predictions:0')
            if threshold is not None:
                predictions = tf.where(tf.reduce_max(logits, axis=-1) > threshold,
                                       initial_student_predictions,
                                       tf.ones_like(initial_student_predictions) * map_misc)
            else:
                predictions = initial_student_predictions
        color_map = tf.constant(colormap_)
        labels_colored = tf.gather(color_map, predictions, axis=0)

        init = tf.initializers.global_variables()
        # TODO: can we remove load_image from this?
        client = {'graph': client_graph, 'image_pl': image_pl, 'load_image': load_image, 'predictions': predictions,
                  'is_inference': is_inference, 'init': init, 'output_colored': labels_colored}
    return client


def create_client_temp(meta_dir_or_bytes, class_weights=None, threshold=None, map_misc=0):
    colormap_ = colormap()
    if class_weights is not None:
        class_weights = np.where(class_weights == 1)[0]
        colormap_ = np.take(colormap_, class_weights, axis=0)
    client_graph = tf.Graph()
    with client_graph.as_default():
        print('Loading the client graph from the meta')
        if isinstance(meta_dir_or_bytes, str):
            tf.train.import_meta_graph('%s.meta' % meta_dir_or_bytes)
        elif isinstance(meta_dir_or_bytes, bytes):
            tf.train.import_meta_graph(meta_dir_or_bytes)
        else:
            raise ValueError('Expected either file directory string or bytes for the meta')
        str_prepend = ''
        if 'student_logits' in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            str_prepend = 'student_'
        image_pl = tf.get_default_graph().get_tensor_by_name('features_input:0')
        label_pl = tf.get_default_graph().get_tensor_by_name('labels_input:0')
        fill_input_buffer = tf.get_default_graph().get_operation_by_name('fill_input_buffer')
        logits = tf.get_default_graph().get_tensor_by_name(str_prepend + 'logits:0')
        if class_weights is not None:
            if threshold is not None:
                student_probs = tf.nn.softmax(logits)
                student_reduced_probs = tf.gather(student_probs, class_weights, axis=-1)
                initial_student_predictions = tf.argmax(student_reduced_probs, axis=-1, output_type=tf.int32)
                predictions = tf.where(tf.reduce_max(student_reduced_probs, axis=-1) > threshold,
                                       initial_student_predictions,
                                       tf.ones_like(initial_student_predictions) * map_misc)
            else:
                student_reduced_logits = tf.gather(logits, class_weights, axis=-1)
                predictions = tf.argmax(student_reduced_logits, axis=-1, output_type=tf.int32)

        else:
            initial_student_predictions = tf.get_default_graph().get_tensor_by_name(str_prepend + 'predictions:0')
            if threshold is not None:
                predictions = tf.where(tf.reduce_max(logits, axis=-1) > threshold,
                                       initial_student_predictions,
                                       tf.ones_like(initial_student_predictions) * map_misc)
            else:
                predictions = initial_student_predictions
        color_map = tf.constant(colormap_)
        labels_colored = tf.gather(color_map, predictions, axis=0)

        init = tf.initializers.global_variables()
        # TODO: can we remove load_image from this?
        client = {'graph': client_graph, 'image_pl': image_pl, 'load_image': fill_input_buffer,
                  'predictions': predictions,
                  'init': init, 'output_colored': labels_colored, 'label_pl': label_pl}
    return client


if __name__ == '__main__':
    create_student_v3_test()
