import functools
import tensorflow as tf

def accuracy():
    return tf.keras.metrics.Accuracy()

def precision():
    return tf.keras.metrics.Precision()

def recall():
    return tf.keras.metrics.Recall()

def roc():
    return tf.keras.metrics.Accuracy()

def f_score():

    return tf.keras.metrics.Accuracy()

def rmse():
    return tf.keras.metrics.RootMeanSquaredError()

def mae():
    return tf.keras.metrics.MeanAbsoluteError()
