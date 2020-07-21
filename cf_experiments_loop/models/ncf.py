"""
Neural Collaborative Filtering Model architecture
"""

import tensorflow as tf

MF_DIM = 8


def mf_slice_fn(input_array):
    """
    :param input_array:
    :return:
    """
    input_array = tf.squeeze(input_array, [1])
    return input_array[:, :MF_DIM]


def mlp_slice_fn(input_array):
    """
    :param input_array:
    :return:
    """
    input_array = tf.squeeze(input_array, [1])
    return input_array[:, MF_DIM:]


def ncf_model(users_number: int, items_number: int,
              model_layers=[64, 32, 16, 8],
              rating_column='rating'):
    """
    :param users_number:
    :param items_number:
    :param model_layers:
    :param rating_column:
    :return:
    """

    user_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_input')

    # Initializer for embedding layers
    embedding_initializer = 'glorot_uniform'

    embedding_user = tf.keras.layers.Embedding(
        input_dim=users_number + 1,
        output_dim=int(model_layers[0] / 2),
        name='user_embedding',
        embeddings_initializer=embedding_initializer,
        embeddings_regularizer=tf.keras.regularizers.l2(),
        input_length=1
    )(user_input)
    embedding_item = tf.keras.layers.Embedding(
        input_dim=items_number + 1,
        output_dim=int(model_layers[0] / 2),
        name='item_embedding',
        embeddings_initializer=embedding_initializer,
        embeddings_regularizer=tf.keras.regularizers.l2(),
        input_length=1
    )(item_input)

    # GMF part
    mf_user_latent = tf.keras.layers.Lambda(
        mf_slice_fn, name="embedding_user_mf"
    )(embedding_user)
    mf_item_latent = tf.keras.layers.Lambda(
        mf_slice_fn, name="embedding_item_mf"
    )(embedding_item)

    # MLP part
    mlp_user_latent = tf.keras.layers.Lambda(
        mlp_slice_fn, name="embedding_user_mlp"
    )(embedding_user)
    mlp_item_latent = tf.keras.layers.Lambda(
        mlp_slice_fn, name="embedding_item_mlp"
    )(embedding_item)

    # Element-wise multiply
    mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])

    # Concatenation of two latent features
    mlp_vector = tf.keras.layers.concatenate([mlp_user_latent, mlp_item_latent])

    num_layer = len(model_layers)  # Number of layers in the MLP
    for layer in range(1, num_layer):
        model_layer = tf.keras.layers.Dense(
            model_layers[layer],
            kernel_regularizer=tf.keras.regularizers.l2(),
            activation='relu')
        mlp_vector = model_layer(mlp_vector)

    # Concatenate GMF and MLP parts
    predict_vector = tf.keras.layers.concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    logits = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer='lecun_uniform',
        name=rating_column
    )(predict_vector)

    # Print model topology.
    model = tf.keras.models.Model([user_input, item_input], logits)

    model.summary()

    return model
