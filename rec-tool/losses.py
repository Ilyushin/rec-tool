"""
Loss functions
"""
import tensorflow as tf


def sparse_categorical_crossentropy():
    """
    :return:
    """
    return tf.keras.losses.SparseCategoricalCrossentropy(
        reduction="sum", from_logits=True
    )


def mean_squared_error():
    """
    :return:
    """
    return tf.keras.losses.MeanSquaredError()
