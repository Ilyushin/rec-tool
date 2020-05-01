import numpy as np
import tensorflow.compat.v1 as tf
from cf_experiments_loop.metrics import mae
from cf_experiments_loop.metrics import rmse
from cf_experiments_loop.models.utils.config import Config
from cf_experiments_loop.models.svdpp import SVDPP
from cf_experiments_loop.models.svdpp import SVD
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Note that x is a 2D numpy array,
# x[i, :] contains the user-item pair, and y[i] is the corresponding rating.


def train_svd(
        data=None,
        users_number=None,
        items_number=None,
        batch_size=64,
        epoch=100,
        model_dir=None):

    x, y = [data.users, data.items], data.ratings
    x_train, x_test, y_train, y_test = train_test_split([data.users, data.items], data.ratings, test_size=0.1)

    config = Config()
    config.num_users = users_number
    config.num_items = items_number
    config.min_value = np.min(y)
    config.max_value = np.max(y)

    with tf.Session() as sess:
        # For SVD++ algorithm, if `dual` is True, then the dual term of items'
        # implicit feedback will be added into the original SVD++ algorithm.
        model = SVDPP(config, sess, dual=False)
        model.train(x_train, y_train, validation_data=(
            x_test, y_test), epochs=epoch, batch_size=batch_size)

        y_pred = model.predict(x_test)
        print('rmse: {}, mae: {}'.format(rmse(y_test, y_pred), mae(y_test, y_pred)))

        # Save model
        model = model.save_model(model_dir)
    return model
