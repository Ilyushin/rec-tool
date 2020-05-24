import tensorflow as tf
import numpy as np


def mf(users_number: int, items_number: int):

    latent_dim, max_rating, min_rating, regs = 10, 5, 1, [0, 0]

    # define placeholder.
    user_id_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_id_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_input')

    # define embedding size and layers.

    user_embedding = tf.keras.layers.Embedding(output_dim=latent_dim, input_dim=users_number + 1,
                                               input_length=1, name='user_embedding',
                                               embeddings_regularizer=tf.keras.regularizers.l2())(user_id_input)

    item_embedding = tf.keras.layers.Embedding(output_dim=latent_dim, input_dim=items_number + 1,
                                               input_length=1, name='item_embedding',
                                               embeddings_regularizer=tf.keras.regularizers.l2())(item_id_input)

    x = tf.keras.layers.Concatenate()([user_embedding, item_embedding])
    x = tf.keras.layers.Dropout(0.05)(x)

    x = tf.keras.layers.Dense(10, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(1, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

    model = tf.keras.models.Model(inputs=[user_id_input, item_id_input], outputs=x)

    return model


# For SVD ++
def _get_implicit_feedback(x, num_users, num_items, dual):
    """Gets implicit feedback from (users, items) pair.

    Args:
        x: A numpy array of shape `(samples, 2)`.
        num_users: An integer, total number of users.
        num_items: An integer, total number of items.
        dual: A bool, deciding whether returns the
            dual term of implicit feedback of items.

    Returns:
        A dictionary that is the sparse format of implicit
            feedback of users, if dual is true.
        A tuple of dictionarys that are the sparse format of
            implicit feedback of users and items, otherwise.
    """

    if not dual:
        N = [[] for u in range(num_users)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)

        return N
    else:
        N = [[] for u in range(num_users)]
        H = [[] for u in range(num_items)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)
            H[i].append(u)

        return N, H


def _convert_to_sparse_format(x):
    """Converts a list of lists into sparse format.

    Args:
        x: A list of lists.

    Returns:
        A dictionary that contains three fields, which are
            indices, values, and the dense shape of sparse matrix.
    """

    sparse = {
        'indices': [],
        'values': []
    }

    for row, x_i in enumerate(x):
        for col, x_ij in enumerate(x_i):
            sparse['indices'].append((row, col))
            sparse['values'].append(x_ij)

    max_col = np.max([len(x_i) for x_i in x]).astype(np.int32)

    sparse['dense_shape'] = (len(x), max_col)

    return sparse


def _create_implicit_feedback(implicit_feedback, dual=False):
    """Returns the (tuple of) sparse tensor(s) of implicit feedback.
    """
    with tf.variable_scope('implicit_feedback'):
        if not dual:
            N = tf.SparseTensor(**implicit_feedback)

            return N
        else:
            N = tf.SparseTensor(**implicit_feedback[0])
            H = tf.SparseTensor(**implicit_feedback[1])

            return N, H