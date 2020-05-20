import shutil
from time import time
import tensorflow as tf
import numpy as np
import pandas as pd
from signal_transformation import helpers
from cf_experiments_loop.metrics import auc, recall, mse
from cf_experiments_loop.ml_flow.ml_flow import log_to_mlflow


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
        learning_rate=0.001,
        use_learning_rate_schedule=False,
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
        optimizer=optimizer,
        metrics=metrics
    )

    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=learning_rate,
        mode='auto'
    )

    callbacks = [tensorboard_callback]
    if use_learning_rate_schedule:
        callbacks.append(reduce_lr_callback)

    # TODO: Fit for bpr and vae models with preprocessing
    history_train = model.fit(
        [train_data.user_id, train_data.item_id],
        train_data.rating,
        batch_size=batch_size,
        epochs=epoch,
        callbacks=callbacks,
        verbose=1
    )

    helpers.create_dir(model_dir)
    model.save(model_dir, save_format='tf')

    history_eval = model.evaluate([test_data.user_id, test_data.item_id], test_data.rating)

    print('Train loss:', history_train.history['loss'][len(history_train.history['loss']) - 1])
    print('Eval loss:', history_eval[0])

    return history_train, history_eval
