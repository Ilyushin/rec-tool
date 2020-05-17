import shutil
import tensorflow as tf
from signal_transformation import helpers
from .grid_search import fit_grid_search


def train_model(
        train_data=None,
        test_data=None,
        users_number=None,
        items_number=None,
        model_fn=None,
        loss_fn=None,
        metrics_fn=None,
        batch_size=None,
        epoch=None,
        model_dir=None,
        log_dir=None,
        clear=False,
        grid_search=False,
):

    if clear:
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.rmtree(log_dir, ignore_errors=True)

    model = model_fn(users_number=users_number, items_number=items_number)
    loss = loss_fn()
    metrics = [metric_fn() for metric_fn in metrics_fn]

    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )

    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # TODO: Fit for bpr and vae models with preprocessing
    # grid search fit

    if grid_search:
        best_score, best_param, gs_results = fit_grid_search(batch_size=batch_size,
                                                             epoch=epoch,
                                                             model=model,
                                                             features=[train_data.user_id, train_data.item_id],
                                                             targets=train_data.rating)
        # fit best parameters
        history_train = model.fit(
            [train_data.user_id, train_data.item_id],
            train_data.rating,
            batch_size=best_param['batch_size'],
            epochs=best_param['epochs'],
            callbacks=[tensorboard_callback],
            verbose=1
        )

    else:
        # fit best parameters
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
