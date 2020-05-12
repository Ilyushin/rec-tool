import tensorflow as tf
import pandas as pd


def vaesf_preprocess(users: list, items: list, ratings: list):
    """
    :param users: list of users
    :param items: list of items
    :param ratings: list of ratings
    :return: sparse matrix
    """
    df = pd.DataFrame({'user_id': users, 'item_id': items, 'ratings': ratings})

    users_items_matrix_df = df.pivot(index='user_id',
                                     columns='content_id',
                                     values='view').fillna(0)
    return users_items_matrix_df


# Autoencoder
def vaecf(users_number: int, items_number: int):

    input_layer = tf.keras.layers.Input(shape=(items_number,), name='UserScore')

    enc = tf.keras.layers.Dense(512, activation='selu', name='EncLayer1')(input_layer)

    lat_space = tf.keras.layers.Dense(256, activation='selu', name='LatentSpace')(enc)
    lat_space = tf.keras.layers.Dropout(0.8, name='Dropout')(lat_space) # Dropout

    dec = tf.keras.layers.Dense(512, activation='selu', name='DecLayer1')(lat_space)

    output_layer = tf.keras.layers.Dense(items_number, activation='linear', name='UserScorePred')(dec)

    # this model maps an input to its reconstruction
    model = tf.keras.models.Model(input_layer, output_layer)

    return model
