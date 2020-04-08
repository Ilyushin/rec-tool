import tensorflow as tf


def train_model(
        train_data=None,
        test_data=None,
        users_number=None,
        items_number=None,
        model_fn=None,
        loss_fn=None,
        metrics_fn=None,
        batch_size=64,
        epoch=10
):
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

    history = model.fit(
        [train_data.user_id, train_data.item_id],
        train_data.rating,
        epochs=5,
        verbose=1
    )

    return None