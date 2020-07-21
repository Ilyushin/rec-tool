import tensorflow as tf
import pandas as pd
import numpy as np


def vaeсf_preprocessing(users: list, items: list, ratings: list):
    """
    :param users: list of users
    :param items: list of items
    :param ratings: list of ratings
    :return: sparse matrix
    """
    data = pd.DataFrame({'user_id': users, 'item_id': items, 'ratings': ratings})

    users_items_matrix_df = data.pivot(index='user_id',
                                       columns='content_id',
                                       values='view').fillna(0)
    return users_items_matrix_df


# Autoencoder
def vaecf(users_number: int, items_number: int):
    """
    :param users_number:
    :param items_number:
    :return:
    """

    input_layer = tf.keras.layers.Input(shape=(items_number,), name='UserScore')

    enc = tf.keras.layers.Dense(512, activation='selu', name='EncLayer1')(input_layer)

    lat_space = tf.keras.layers.Dense(256, activation='selu', name='LatentSpace')(enc)
    lat_space = tf.keras.layers.Dropout(0.8, name='Dropout')(lat_space)

    dec = tf.keras.layers.Dense(512, activation='selu', name='DecLayer1')(lat_space)

    output_layer = tf.keras.layers.Dense(items_number,
                                         activation='linear',
                                         name='UserScorePred')(dec)

    # this model maps an input to its reconstruction
    model = tf.keras.models.Model(input_layer, output_layer)

    return model


def vae_preprocess_data(data):
    """
    :param data: pd.DataFrame with columns ['user_id', 'item_id', 'rating']
    :return:
    """

    unique_users = list(set(data.user_id))
    unique_items = list(set(data.item_id))

    vae_sparse_matrix = np.zeros((len(unique_users), len(unique_items)))

    for i in range(1, len(data.user_id)):
        row_index = unique_users.index(list(data.user_id)[i])
        col_index = unique_items.index(list(data.item_id)[i])
        vae_sparse_matrix[row_index, col_index] = list(data.rating)[i]

    return vae_sparse_matrix
