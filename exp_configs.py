import numpy as np


# If you want to add a video, you have to return its number of classes in num_classes,
# its important classes in class_weights and its length in test_length. If the video is to be used with coco labeling
# but its original labels are from PASCAL VOC, add it to is_coco.

def num_classes(experiment_number):
    if experiment_number in [12, 13, 14, 15, 17, 19, 21, 22, 23, 24, 25]:
        return 19
    elif experiment_number in [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                               49, 50, 51, 52, 53, 54]:
        return 21
    else:
        raise ValueError('Experiment %d not configured' % experiment_number)


def class_weights(experiment_number):
    # Cityscapes
    # 0:  'road'
    # 1:  'sidewalk'
    # 2:  'building'
    # 3:  'wall'
    # 4:  'fence'
    # 5:  'pole'
    # 6:  'traffic light'
    # 7:  'traffic sign'
    # 8:  'vegetation'
    # 9:  'terrain'
    # 10: 'sky'
    # 11: 'person'
    # 12: 'rider'
    # 13: 'car'
    # 14: 'truck'
    # 15: 'bus'
    # 16: 'train'
    # 17: 'motorcycle'
    # 18: 'bicycle'
    if experiment_number == 0:
        class_weights_exp = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    # Outdoor Scenes
    elif experiment_number == 12:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 13:
        class_weights_exp = np.array(
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 14:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 15:
        class_weights_exp = np.array(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 17:
        class_weights_exp = np.array(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 19:
        class_weights_exp = np.array(
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 21:
        class_weights_exp = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    # A2D2
    elif experiment_number == 22:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 23:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 24:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    # Cityscapes
    elif experiment_number == 25:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    # coco with 21 classes:
    # 0 = background
    # 1 = aeroplane
    # 2 = bicycle
    # 3 = bird
    # 4 = boat
    # 5 = bottle
    # 6 = bus
    # 7 = car
    # 8 = cat
    # 9 = chair
    # 10 = cow
    # 11 = dining table
    # 12 = dog
    # 13 = horse
    # 14 = motorbike
    # 15 = person
    # 16 = potted plant
    # 17 = sheep
    # 18 = sofa
    # 19 = train
    # 20 = tv / monitor
    # LVS
    elif experiment_number == 26:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 27:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 28:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 29:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 30:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 31:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 32:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 33:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 34:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 35:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 36:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 37:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 39:
        class_weights_exp = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 40:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 41:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 42:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 43:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 44:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 45:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 46:
        class_weights_exp = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 47:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 48:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 49:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 50:
        class_weights_exp = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 51:
        class_weights_exp = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 52:
        class_weights_exp = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 53:
        class_weights_exp = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 54:
        class_weights_exp = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    else:
        raise ValueError('Experiment %d not configured' % experiment_number)
    return np.reshape(class_weights_exp, (num_classes(experiment_number), 1))


def test_length(experiment_number):
    if experiment_number == 12:
        # 30 fps
        length = 900
    elif experiment_number == 13:
        # 30 fps
        length = 420
    elif experiment_number == 14:
        # 30 fps
        length = 810
    elif experiment_number == 15:
        # 30 fps
        length = 900
    elif experiment_number == 17:
        # 30 fps
        length = 900
    elif experiment_number == 19:
        # 30 fps
        length = 900
    elif experiment_number == 21:
        # 30 fps
        length = 800
    elif experiment_number == 22:
        # 30 fps
        length = 520
    elif experiment_number == 23:
        # 30 fps
        length = 900
    elif experiment_number == 24:
        # 30 fps
        length = 740
    elif experiment_number == 25:
        # 30 fps
        length = 2790
    elif experiment_number == 26:
        # 30 fps, 30000 frames
        length = 1000
    elif experiment_number == 27:
        # 30 fps, 30000 frames
        length = 1000
    elif experiment_number == 28:
        # 25 fps, 30000 frames
        length = 1200
    elif experiment_number == 29:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 30:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 31:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 32:
        # 59.94 fps, 30000 frames
        length = 500
    elif experiment_number == 33:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 34:
        # 29.85 fps, 30000 frames
        length = 1000
    elif experiment_number == 35:
        # 30 fps, 30000 frames
        length = 1000
    elif experiment_number == 36:
        # 25 fps, 29817 frame
        length = 1190
    elif experiment_number == 37:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 39:
        # 50 fps, 30000 frames
        length = 600
    elif experiment_number == 40:
        # 29.96 fps, 30000 frames
        length = 1000
    elif experiment_number == 41:
        # 23.98 fps, 30000 frames
        length = 1250
    elif experiment_number == 42:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 43:
        # 59.94 fps, 30000 frames
        length = 500
    elif experiment_number == 44:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 45:
        # 59.94 fps, 30000 frames
        length = 500
    elif experiment_number == 46:
        # 60 fps, 30000 frames
        length = 500
    elif experiment_number == 47:
        # 12 fps, 21441 frame
        length = 1780
    elif experiment_number == 48:
        # 25 fps, 30000 frames
        length = 1200
    elif experiment_number == 49:
        # 30 fps, 30000 frames
        length = 1000
    elif experiment_number == 50:
        # 30 fps, 30000 frames
        length = 1000
    elif experiment_number == 51:
        # 30 fps, 30000 frames
        length = 1000
    elif experiment_number == 52:
        # 29.89 fps, 30000 frames
        length = 1000
    elif experiment_number == 53:
        # 29.97 fps, 30000 frames
        length = 1000
    elif experiment_number == 54:
        # 29.97 fps, 30000 frames
        length = 1000
    else:
        raise ValueError('Experiment %d not configured' % experiment_number)
    return length


def coco_class_converter():
    return np.array([0, 15, 2, 7, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 3, 0, 12, 13, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, ], dtype=np.int32)


def is_coco(experiment_number):
    return experiment_number in [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                 49, 50, 51, 52, 53, 54]
