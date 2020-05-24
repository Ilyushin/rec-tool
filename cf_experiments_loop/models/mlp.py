from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge, Dropout, Dense, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf


def mlp(users_number: int, items_number: int):
    latent_dim = 10

    # Define inputs
    movie_input = Input(shape=[1],name='movie-input')
    user_input = Input(shape=[1], name='user-input')

    # MLP Embeddings
    movie_embedding_mlp = Embedding(items_number + 1, latent_dim, name='movie-embedding-mlp')(movie_input)
    movie_vec_mlp = Flatten(name='flatten-movie-mlp')(movie_embedding_mlp)

    user_embedding_mlp = Embedding(users_number + 1, latent_dim, name='user-embedding-mlp')(user_input)
    user_vec_mlp = Flatten(name='flatten-user-mlp')(user_embedding_mlp)

    # MF Embeddings
    movie_embedding_mf = Embedding(items_number + 1, latent_dim, name='movie-embedding-mf')(movie_input)
    movie_vec_mf = Flatten(name='flatten-movie-mf')(movie_embedding_mf)

    user_embedding_mf = Embedding(users_number + 1, latent_dim, name='user-embedding-mf')(user_input)
    user_vec_mf = Flatten(name='flatten-user-mf')(user_embedding_mf)

    # MLP layers
    concat = tf.keras.layers.concatenate([movie_vec_mlp, user_vec_mlp], mode='concat', name='concat')
    concat_dropout = Dropout(0.2)(concat)
    fc_1 = Dense(100, name='fc-1', activation='relu')(concat_dropout)
    fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
    fc_1_dropout = Dropout(0.2)(fc_1_bn)
    fc_2 = Dense(50, name='fc-2', activation='relu')(fc_1_dropout)
    fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
    fc_2_dropout = Dropout(0.2)(fc_2_bn)

    # Prediction from both layers
    pred_mlp = Dense(10, name='pred-mlp', activation='relu')(fc_2_dropout)
    pred_mf = tf.keras.layers.Dot([movie_vec_mf, user_vec_mf], mode='dot', name='pred-mf')
    combine_mlp_mf = tf.keras.layers.concatenate([pred_mf, pred_mlp], mode='concat', name='combine-mlp-mf')

    # Final prediction
    result = Dense(1, name='result', activation='relu')(combine_mlp_mf)

    model = Model([user_input, movie_input], result)
    return model
