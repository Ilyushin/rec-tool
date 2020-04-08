import tensorflow as tf


def sparse_categorical_crossentropy(labels, embeddings, margin=0.5):
    return tf.keras.losses.SparseCategoricalCrossentropy(
        reduction="sum", from_logits=True
    )

def mean_squared_error():
    return tf.keras.losses.MeanSquaredError()
