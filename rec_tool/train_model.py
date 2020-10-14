"""
Train method for Collaborative Filtering models
"""
import shutil
import tensorflow as tf
import numpy as np
from sklearn.metrics import ndcg_score
from cf_experiments_loop.common import fn
from cf_experiments_loop.models.config import Config
from cf_experiments_loop.models.svdpp import mse, rmse, mae
from cf_experiments_loop.models.bpr_model import bpr_preprocess_data
from cf_experiments_loop.models.vae import vae_preprocess_data

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1:], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


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
        batch_size=64,
        epoch=10,
):

    """
    :param train_data:
    :param test_data:
    :param users_number:
    :param items_number:
    :param model_fn:
    :param loss_fn:
    :param metrics_fn:
    :param model_dir:
    :param log_dir:
    :param clear:
    :param optimizer:
    :param batch_size:
    :param epoch:
    :return:
    """

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
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    if model_fn.__name__ == 'bpr':

        bpr_train_data, bpr_test_data = bpr_preprocess_data(users=train_data.user_id,
                                                            items=train_data.item_id,
                                                            rating=train_data.rating)

        history_train = model.fit(
            bpr_train_data,
            np.ones(users_number),
            batch_size=batch_size,
            epochs=epoch,
            callbacks=[tensorboard_callback, early_stopping_callback],
            verbose=1
        )

        history_eval = model.evaluate(bpr_test_data)

        ndcg = ndcg_score(y_true=np.asarray([list(test_data.rating)]),
                          y_score=model.predict([test_data.user_id,
                                                 test_data.item_id]).reshape(1, -1))

        print('Train loss:', history_train.history['loss'][len(history_train.history['loss']) - 1])
        print('Eval loss:', history_eval[0])

    elif model_fn.__name__ == 'vaecf':

        vae_train_data = vae_preprocess_data(train_data)

        vae_test_data = vae_preprocess_data(test_data)

        history_train = model.fit(
            vae_train_data,
            vae_train_data,
            batch_size=batch_size,
            epochs=epoch,
            callbacks=[tensorboard_callback, early_stopping_callback],
            verbose=1
        )

        history_eval = model.evaluate(vae_test_data, vae_test_data)

        ndcg = ndcg_score(y_true=np.asarray([list(test_data.rating)]),
                          y_score=model.predict([test_data.user_id,
                                                 test_data.item_id]).reshape(1, -1))

        print('Train loss:', history_train.history['loss'][len(history_train.history['loss']) - 1])
        print('Eval loss:', history_eval[0])

    else:

        history_train = model.fit(
            [train_data.user_id, train_data.item_id],
            train_data.rating,
            batch_size=batch_size,
            epochs=epoch,
            callbacks=[tensorboard_callback, early_stopping_callback],
            verbose=1
        )

        history_eval = model.evaluate([test_data.user_id, test_data.item_id], test_data.rating)

        ndcg = ndcg_score(y_true=np.asarray([list(test_data.rating)]),
                          y_score=model.predict([test_data.user_id,
                                                 test_data.item_id]).reshape(1, -1))

        print('Train loss:', history_train.history['loss'][len(history_train.history['loss'])-1])
        print('Eval loss:', history_eval[0])

    return model, history_train, history_eval, ndcg


def train_svd(
        train_data=None,
        test_data=None,
        users_number=None,
        items_number=None,
        model_fn=None,
        batch_size=64,
        epoch=10):

    """
    :param train_data:
    :param test_data:
    :param users_number:
    :param items_number:
    :param model_fn:
    :param batch_size:
    :param epoch:
    :return:
    """

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
                    train_data['rating'].to_numpy(),
                    validation_data=(test_data[['user_id', 'item_id']].to_numpy(),
                                     test_data['rating'].to_numpy()),
                    epochs=epoch,
                    batch_size=batch_size)

        # evaluate with metrics
        y_pred = model.predict(test_data[['user_id', 'item_id']].to_numpy())

        # calculate ndcg metric
        ndcg = ndcg_score(y_true=np.asarray([list(test_data.rating)]),
                          y_score=model.predict(test_data[['user_id',
                                                           'item_id']].to_numpy().reshape(1, -1)))

        # collect metrics
        history_eval = {
            'mse': mse(test_data['rating'].to_numpy(), y_pred),
            'rmse': rmse(test_data['rating'].to_numpy(), y_pred),
            'mae': mae(test_data['rating'].to_numpy(), y_pred),
            'ndcg': ndcg
        }

    return model, history_eval


# help function to not repeat the code
def train_both_types(model_path: str,
                     metric_names: list,
                     train_data=None,
                     test_data=None,
                     users_number=None,
                     items_number=None,
                     loss_fn=None,
                     metrics_fn=None,
                     model_dir=None,
                     log_dir=None,
                     clear=False,
                     batch_size=64,
                     epoch=10
                     ):

    if model_path == 'rec_tool.models.svdpp.svdpp':
        model, history_eval = train_svd(
            train_data=train_data,
            test_data=test_data,
            users_number=users_number,
            items_number=items_number,
            model_fn=fn(model_path),
            batch_size=batch_size,
            epoch=epoch
        )

        metrics = history_eval

    else:

        model, history_train, history_eval, ndcg = train_model(
            train_data=train_data,
            test_data=test_data,
            users_number=users_number,
            items_number=items_number,
            model_fn=fn(model_path),
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            model_dir=model_dir,
            log_dir=log_dir,
            clear=clear,
            batch_size=batch_size,
            epoch=epoch
        )

        # create dictionary for metrics
        metrics = {metric.split('.')[-1]:
                   history_eval[metric_names.index(metric) + 1]
                   for metric in metric_names}

        metrics['ndcg'] = ndcg
    return model, metrics
