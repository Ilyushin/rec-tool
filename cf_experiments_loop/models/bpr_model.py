"""
Bayesian Personalized Ranking model
"""

import tensorflow as tf
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import train_test_split


@tf.function
def identity_loss(_, y_pred):
    """
    :param _:
    :param y_pred:
    :return:
    """
    return tf.math.reduce_mean(y_pred)


@tf.function
def bpr_triplet_loss(data):
    """
    Calculate triplet loss - as higher the difference between positive interactions
    and negative interactions as better

    :param x: x contains the user input, positive item input, negative item input
    :return:
    """
    positive_item_latent, negative_item_latent, user_latent = data

    positive_interactions = tf.math.reduce_sum(tf.math.multiply(user_latent,
                                                                positive_item_latent),
                                               axis=-1,
                                               keepdims=True)
    negative_interactions = tf.math.reduce_sum(tf.math.multiply(user_latent,
                                                                negative_item_latent),
                                               axis=-1,
                                               keepdims=True)

    return tf.math.subtract(tf.constant(1.0),
                            tf.sigmoid(tf.math.subtract(positive_interactions,
                                                        negative_interactions)))


def bpr(users_number: int, items_number: int):
    """
    Build a model for Bayesian personalized ranking

    :param users_number: a number of the unique users
    :param items_number: a number of the unique movies
    :return: Model
    """
    latent_dim = 100

    positive_item_input = tf.keras.layers.Input((1,), name='positive_item_input')
    negative_item_input = tf.keras.layers.Input((1,), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = tf.keras.layers.Embedding(
        items_number, latent_dim, name='item_embedding', input_length=1)

    user_input = tf.keras.layers.Input((1,), name='user_input')

    positive_item_embedding = tf.keras.layers.Flatten()(item_embedding_layer(
        positive_item_input))
    negative_item_embedding = tf.keras.layers.Flatten()(item_embedding_layer(
        negative_item_input))
    user_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
        users_number, latent_dim, name='user_embedding', input_length=1)(
        user_input))

    triplet_loss = tf.keras.layers.Lambda(bpr_triplet_loss)([positive_item_embedding,
                                                             negative_item_embedding,
                                                             user_embedding])

    model = tf.keras.models.Model(inputs=[positive_item_input,
                                          negative_item_input,
                                          user_input],
                                  outputs=triplet_loss)

    return model


def _build_interaction_matrix(rows, cols, data, threshold=3.0):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)
    users, items, ratings = list(data['user_id']), list(data['item_id']), list(data['rating'])

    for i in range(len(users)):

        # Let's assume only really good things are positives
        if ratings[i] >= threshold:
            mat[users[i], items[i]] = 1.0

    return mat.tocoo()


def get_triplets(mat):
    """
    :param mat:
    :return:
    """
    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))


def get_data(data):
    """
    Return (train_interactions, test_interactions).
    """

    uids = set(data['user_id'])
    iids = set(data['item_id'])

    rows = max(uids) + 1
    cols = max(iids) + 1

    train, test = train_test_split(data)

    return (_build_interaction_matrix(rows, cols, train),
            _build_interaction_matrix(rows, cols, test))


def bpr_preprocess_data(users: list, items: list, rating: list):
    """
    :param users:
    :param items:
    :param rating:
    :return:
    """
    dataset = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': rating})
    train, test = get_data(dataset)
    return train, test
