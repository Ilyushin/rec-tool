import shutil
import tensorflow as tf
import numpy as np
from signal_transformation import helpers
from cf_experiments_loop.models.config import Config
from cf_experiments_loop.models.svdpp import mse, rmse, mae


def train_model(
        train_data=None,
        test_data=None,
        users_number=None,
        items_number=None,
        model_fn=None,
        loss_fn=None,
        metrics_fn=None,
        model_dir=None,
        log_dir=None,
        clear=False,
        optimizer=None,
        batch_size=64,
        epoch=10,
):

    if clear:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)

    model = model_fn(users_number=users_number, items_number=items_number)
    loss = loss_fn()
    metrics = [metric_fn() for metric_fn in metrics_fn]

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=metrics
    )

    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # TODO: Fit for bpr and vae models with preprocessing
    history_train = model.fit(
        [train_data.user_id, train_data.item_id],
        train_data.rating,
        batch_size=batch_size,
        epochs=epoch,
        callbacks=[tensorboard_callback],
        verbose=1
    )

    helpers.create_dir(model_dir)
    model.save(model_dir, save_format='tf')

    history_eval = model.evaluate([test_data.user_id, test_data.item_id], test_data.rating)

    print('Train loss:', history_train.history['loss'][len(history_train.history['loss'])-1])
    print('Eval loss:', history_eval[0])

    return history_train, history_eval


def train_svd(
        train_data=None,
        test_data=None,
        users_number=None,
        items_number=None,
        model_fn=None,
        loss_fn=None,
        metrics_fn=None,
        model_dir=None,
        log_dir=None,
        clear=False,
        optimizer=None,
        batch_size=64,
        epoch=10):

    config = Config()
    config.num_users = users_number
    config.num_items = items_number
    config.min_value = np.min(train_data.rating)
    config.max_value = np.max(train_data.rating)

    with tf.compat.v1.Session() as sess:

        # For SVD++ algorithm, if `dual` is True, then the dual term of items'
        # implicit feedback will be added into the original SVD++ algorithm.
        model = model_fn(config, sess, dual=False)
        model.train(train_data[['user_id', 'item_id']].to_numpy(),
                    np.reshape(train_data['rating'].to_numpy(), (-1, 1)),
                    validation_data=(test_data[['user_id', 'item_id']].to_numpy(),
                                     np.reshape(test_data['rating'].to_numpy(), (-1, 1))),
                    epochs=epoch,
                    batch_size=batch_size)

        # save model
        model = model.save_model(model_dir)

        # evaluate with metrics
        y_pred = model.predict(test_data[['user_id', 'item_id']].to_numpy())

        # collect metrics
        history_eval = {'mse': mse(np.reshape(test_data['rating'].to_numpy(), (-1, 1)), y_pred),
                        'rmse': rmse(np.reshape(test_data['rating'].to_numpy(), (-1, 1)), y_pred),
                        'mae': mae(np.reshape(test_data['rating'].to_numpy(), (-1, 1)), y_pred)}

    return _, history_eval
