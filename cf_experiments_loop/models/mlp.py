import tensorflow as tf


def mlp(users_number: int, items_number: int):
    latent_dim = 10

    # Define inputs
    movie_input = tf.keras.layers.Input(shape=[1], name='movie-input')
    user_input = tf.keras.layers.Input(shape=[1], name='user-input')

    # MLP Embeddings
    movie_embedding_mlp = tf.keras.layers.Embedding(items_number + 1, latent_dim, name='movie-embedding-mlp')(movie_input)
    movie_vec_mlp = tf.keras.layers.Flatten(name='flatten-movie-mlp')(movie_embedding_mlp)

    user_embedding_mlp = tf.keras.layers.Embedding(users_number + 1, latent_dim, name='user-embedding-mlp')(user_input)
    user_vec_mlp = tf.keras.layers.Flatten(name='flatten-user-mlp')(user_embedding_mlp)

    # MF Embeddings
    movie_embedding_mf = tf.keras.layers.Embedding(items_number + 1, latent_dim, name='movie-embedding-mf')(movie_input)
    movie_vec_mf = tf.keras.layers.Flatten(name='flatten-movie-mf')(movie_embedding_mf)

    user_embedding_mf = tf.keras.layers.Embedding(users_number + 1, latent_dim, name='user-embedding-mf')(user_input)
    user_vec_mf = tf.keras.layers.Flatten(name='flatten-user-mf')(user_embedding_mf)

    # MLP layers
    concat = tf.keras.layers.Concatenate()([movie_vec_mlp, user_vec_mlp])
    concat_dropout = tf.keras.layers.Dropout(0.2)(concat)
    fc_1 = tf.keras.layers.Dense(100, name='fc-1', activation='relu')(concat_dropout)
    fc_1_bn = tf.keras.layers.BatchNormalization(name='batch-norm-1')(fc_1)
    fc_1_dropout = tf.keras.layers.Dropout(0.2)(fc_1_bn)
    fc_2 = tf.keras.layers.Dense(50, name='fc-2', activation='relu')(fc_1_dropout)
    fc_2_bn = tf.keras.layers.BatchNormalization(name='batch-norm-2')(fc_2)
    fc_2_dropout = tf.keras.layers.Dropout(0.2)(fc_2_bn)

    # Prediction from both layers
    pred_mlp = tf.keras.layers.Dense(10, name='pred-mlp', activation='relu')(fc_2_dropout)
    pred_mf = tf.keras.layers.Dot(axes=1)([movie_vec_mf, user_vec_mf])
    combine_mlp_mf = tf.keras.layers.Concatenate()([pred_mf, pred_mlp])

    # Final prediction
    result = tf.keras.layers.Dense(1, name='result', activation='relu')(combine_mlp_mf)

    model = tf.keras.models.Model([user_input, movie_input], result)
    return model
