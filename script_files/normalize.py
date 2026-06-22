import numpy as np
import tensorflow as tf

def normalize_data(train_data, test_data):

    mean = np.mean(train_data)
    std = np.std(train_data)

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    return train_data, test_data