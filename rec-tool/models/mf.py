"""
Matrix Factorization model
"""
import tensorflow as tf


def matrix_factorisation(users_number: int, items_number: int):
    """
    :param users_number: int: unique users number
    :param items_number: int: unique items number
    :return: MF model
    """

    latent_dim, max_rating, min_rating = 10, 5, 1

    # define placeholder.
    user_id_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_id_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_input')

    # define embedding size and layers.

    user_embedding = tf.keras.layers.Embedding(
        output_dim=latent_dim,
        input_dim=users_number + 1,
        input_length=1,
        name='user_embedding',
        embeddings_regularizer=tf.keras.regularizers.l2())(user_id_input)

    item_embedding = tf.keras.layers.Embedding(
        output_dim=latent_dim,
        input_dim=items_number + 1,
        input_length=1,
        name='item_embedding',
        embeddings_regularizer=tf.keras.regularizers.l2())(item_id_input)

    output = tf.keras.layers.Concatenate()([user_embedding, item_embedding])
    output = tf.keras.layers.Dropout(0.05)(output)

    output = tf.keras.layers.Dense(10, kernel_initializer='he_normal')(output)
    output = tf.keras.layers.Activation('relu')(output)
    output = tf.keras.layers.Dropout(0.5)(output)

    output = tf.keras.layers.Dense(1, activation='relu', kernel_initializer='he_normal')(output)
    output = tf.keras.layers.Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(output)

    model = tf.keras.models.Model(inputs=[user_id_input, item_id_input], outputs=output)

    return model
