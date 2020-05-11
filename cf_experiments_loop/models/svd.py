import tensorflow as tf
import numpy as np


# TODO: SVD ++, TimeSVD ++
def svd(users_number: int, items_number: int):

    latent_dim, max_rating, min_rating, regs = 100, 5, 1, [0, 0]

    # define placeholder.
    user_id_input = tf.keras.layers.Input(shape=[1], name='user')
    item_id_input = tf.keras.layers.Input(shape=[1], name='item')

    # define embedding size and layers.

    user_embedding = tf.keras.layers.Embedding(output_dim=latent_dim, input_dim=users_number,
                                               input_length=1, name='user_embedding',
                                               embeddings_regularizer=tf.keras.regularizers.l2(regs[0]))(user_id_input)
    item_embedding = tf.keras.layers.Embedding(output_dim=latent_dim, input_dim=items_number,
                                               input_length=1, name='item_embedding',
                                               embeddings_regularizer=tf.keras.regularizers.l2(regs[1]))(item_id_input)

    user_bias = tf.keras.layers.Embedding(input_dim=users_number, output_dim=1)(user_id_input)
    item_bias = tf.keras.layers.Embedding(input_dim=items_number, output_dim=1)(item_id_input)

    user_vecs = tf.keras.layers.Reshape([latent_dim])(user_embedding)
    item_vecs = tf.keras.layers.Reshape([latent_dim])(item_embedding)

    # The prediction, which we calculate the loss function with ground truth and optimize.
    y_hat = tf.keras.layers.Dot(1, normalize=False)([user_vecs, item_vecs])
    # add tf.keras.backend.constant(np.mean(targets), shape=[])
    y_hat = tf.keras.layers.Add()([y_hat, user_bias, item_bias])

    x = tf.keras.layers.Activation('sigmoid')(y_hat)
    x = tf.keras.layers.Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = tf.keras.models.Model(inputs=[user_id_input, item_id_input], outputs=x)

    return model