import tensorflow as tf
import pandas as pd
from tqdm import tqdm


def bpr_preprocess_data(users: list, items: list, rating: list, rating_threshold: int):
    """
    :param users: list of users
    :param items: list of items
    :param rating: list of ratings
    :param rating_threshold: number for filtering positive and negative items
    :return: dict
    """
    train_set = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': rating})

    df_triplest = pd.DataFrame(columns=['user_id', 'positive_m_id', 'negative_m_id'])

    data = []
    users_without_data = []

    for user_id in tqdm(train_set.user_id.unique()):
        positive_movies = train_set[(train_set.user_id == user_id) & (train_set.rating > rating_threshold)].item_id.values
        negative_movies = train_set[(train_set.user_id == user_id) & (train_set.rating <= rating_threshold)].item_id.values

        if negative_movies.shape[0] == 0 or positive_movies.shape[0] == 0:
            users_without_data.append(user_id)
            continue

        for positive_movie in positive_movies:
            for negative_movie in negative_movies:
                data.append({'user_id': user_id, 'positive_m_id': positive_movie, 'negative_m_id': negative_movie})

    df_triplest = df_triplest.append(data, ignore_index=True)

    X = {
        'user_input': tf.convert_to_tensor(df_triplest.user_id),
        'positive_item_input': tf.convert_to_tensor(df_triplest.positive_m_id),
        'negative_item_input': tf.convert_to_tensor(df_triplest.negative_m_id)
    }

    return X


@tf.function
def identity_loss(_, y_pred):
    return tf.math.reduce_mean(y_pred)

@tf.function
def bpr_triplet_loss(x: dict):
    """
    Calculate triplet loss - as higher the difference between positive interactions
    and negative interactions as better

    :param x: x contains the user input, positive item input, negative item input
    :return:
    """
    positive_item_latent, negative_item_latent, user_latent = x

    positive_interactions = tf.math.reduce_sum(tf.math.multiply(user_latent, positive_item_latent), axis=-1, keepdims=True)
    negative_interactions = tf.math.reduce_sum(tf.math.multiply(user_latent, negative_item_latent), axis=-1, keepdims=True)

    return tf.math.subtract(tf.constant(1.0), tf.sigmoid(tf.math.subtract(positive_interactions, negative_interactions)))


def bpr(users_number: int, items_number: int):
    """
    Build a model for Bayesian personalized ranking

    :param users_number: a number of the unique users
    :param items_number: a number of the unique movies
    :return: Model
    """
    latent_dim = 100

    user_input = tf.keras.layers.Input((1,), name='user_input')

    positive_item_input = tf.keras.layers.Input((1,), name='positive_item_input')
    negative_item_input = tf.keras.layers.Input((1,), name='negative_item_input')

    # One embedding layer is shared between positive and negative items
    item_embedding_layer = tf.keras.layers.Embedding(items_number + 1, latent_dim, name='item_embedding', input_length=1)

    positive_item_embedding = tf.keras.layers.Flatten()(item_embedding_layer(positive_item_input))
    negative_item_embedding = tf.keras.layers.Flatten()(item_embedding_layer(negative_item_input))

    user_embedding = tf.keras.layers.Embedding(users_number + 1, latent_dim, name='user_embedding', input_length=1)(user_input)
    user_embedding = tf.keras.layers.Flatten()(user_embedding)

    triplet_loss = tf.keras.layers.Lambda(bpr_triplet_loss)([positive_item_embedding,
                                                             negative_item_embedding,
                                                             user_embedding])

    model = tf.keras.models.Model(inputs=[positive_item_input, negative_item_input, user_input], outputs=triplet_loss)

    return model
