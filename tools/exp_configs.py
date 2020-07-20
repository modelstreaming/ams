import numpy as np


def class_weights(experiment_number):
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
    elif experiment_number == 1:
        class_weights_exp = np.array(
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 2:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 3:
        class_weights_exp = np.array(
            [0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
             1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 4:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 5:
        raise ValueError("Not ready")
    elif experiment_number == 6:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 7:
        class_weights_exp = np.array(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 8:
        class_weights_exp = np.array(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 9:
        raise ValueError("Not ready")
    elif experiment_number == 10:
        class_weights_exp = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 11:
        class_weights_exp = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
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
    elif experiment_number == 16:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 17:
        class_weights_exp = np.array(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 18:
        class_weights_exp = np.array(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 19:
        class_weights_exp = np.array(
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 20:
        class_weights_exp = np.array(
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    elif experiment_number == 21:
        class_weights_exp = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
             1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    else:
        raise ValueError("Unknown Experiment")
    return np.reshape(class_weights_exp, (19, 1))


def test_length(experiment_number):
    if 1 < experiment_number <= 9:
        length = 300
    elif experiment_number == 10:
        raise ValueError("Not ready")
    elif experiment_number == 11:
        raise ValueError("Not ready")
    elif experiment_number == 12:
        length = 900
    elif experiment_number == 13:
        length = 420
    elif experiment_number == 14:
        length = 810
    elif experiment_number == 15:
        length = 900
    elif experiment_number == 16:
        length = 900
    elif experiment_number == 17:
        length = 900
    elif experiment_number == 18:
        length = 900
    elif experiment_number == 19:
        length = 900
    elif experiment_number == 20:
        length = 900
    elif experiment_number == 21:
        length = 800
    else:
        raise ValueError("Unknown Experiment")
    return length
