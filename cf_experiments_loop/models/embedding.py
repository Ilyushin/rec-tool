import functools
import tensorflow as tf


def embedding_model(users_number=None, items_number=None):

    # creating book embedding path
    items_input = tf.keras.layers.Input(shape=[1], name="items_input")
    items_embedding = tf.keras.layers.Embedding(items_number + 1, 5, name="items_embeddings")(items_input)
    items_vec = tf.keras.layers.Flatten(name="items_flatten")(items_embedding)

    # creating user embedding path
    users_input = tf.keras.layers.Input(shape=[1], name="user_input")
    users_embedding = tf.keras.layers.Embedding(users_number + 1, 5, name="users_embedding")(users_input)
    users_vec = tf.keras.layers.Flatten(name="users_flatten")(users_embedding)

    # performing dot product and creating model
    prod = tf.keras.layers.Dot(name="dot_product", axes=1)([items_vec, users_vec])
    model = tf.keras.Model([users_input, items_input], prod)

    return model


def embedding_model_test(users_number=None, items_number=None):

    # Let's use a higher latent dimension.
    latent_dim = 10

    item_input = tf.keras.layers.Input(shape=[1], name='item-input')
    item_embedding = tf.keras.layers.Embedding(items_number + 1, latent_dim, name='item-embedding')(item_input)
    item_vec = tf.keras.layers.Flatten(name='movie-flatten')(item_embedding)

    user_input = tf.keras.layers.Input(shape=[1], name='user-input')
    user_embedding = tf.keras.layers.Embedding(users_number + 1, latent_dim, name='user-embedding')(user_input)
    users_vec = tf.keras.layers.Flatten(name='user-flatten')(user_embedding)

    prod = tf.keras.layers.Dot(name="dot_product", axes=1)([item_vec, users_vec])

    model = tf.keras.models.Model([user_input, item_input], prod)

    return model