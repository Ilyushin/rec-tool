import functools
import tensorflow as tf


def embedding_model(users_number=None, items_number=None):

    # creating book embedding path
    items_input = tf.keras.layers.Input(shape=[1], name="items_input")
    items_embedding = tf.keras.layers.Embedding(items_number + 1, 10, name="items_embeddings")(items_input)
    items_vec = tf.keras.layers.Flatten(name="items_flatten")(items_embedding)

    # creating user embedding path
    users_input = tf.keras.layers.Input(shape=[1], name="user_input")
    users_embedding = tf.keras.layers.Embedding(users_number + 1, 10, name="users_embedding")(users_input)
    users_vec = tf.keras.layers.Flatten(name="users_flatten")(users_embedding)

    # performing dot product and creating model
    prod = tf.keras.layers.Dot(name="dot_product", axes=1)([items_vec, users_vec])
    model = tf.keras.Model([users_input, items_input], prod)

    return model
