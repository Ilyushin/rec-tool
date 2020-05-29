import numpy as np
import tensorflow as tff
import tensorflow.compat.v1 as tf
import os
import inspect
from ..metrics import mae, rmse
from cf_experiments_loop.utils.data_utils import BatchGenerator

try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils


def _convert_to_sparse_format(x):
    """Converts a list of lists into sparse format.

    Args:
        x: A list of lists.

    Returns:
        A dictionary that contains three fields, which are
            indices, values, and the dense shape of sparse matrix.
    """

    sparse = {
        'indices': [],
        'values': []
    }

    for row, x_i in enumerate(x):
        for col, x_ij in enumerate(x_i):
            sparse['indices'].append((row, col))
            sparse['values'].append(x_ij)

    max_col = np.max([len(x_i) for x_i in x]).astype(np.int32)

    sparse['dense_shape'] = (len(x), max_col)

    return sparse


def _get_implicit_feedback(x, num_users, num_items, dual):
    """Gets implicit feedback from (users, items) pair.

    Args:
        x: A numpy array of shape `(samples, 2)`.
        num_users: An integer, total number of users.
        num_items: An integer, total number of items.
        dual: A bool, deciding whether returns the
            dual term of implicit feedback of items.

    Returns:
        A dictionary that is the sparse format of implicit
            feedback of users, if dual is true.
        A tuple of dictionarys that are the sparse format of
            implicit feedback of users and items, otherwise.
    """

    if not dual:
        N = [[] for u in range(num_users)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)

        return _convert_to_sparse_format(N)
    else:
        N = [[] for u in range(num_users)]
        H = [[] for u in range(num_items)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)
            H[i].append(u)

        return _convert_to_sparse_format(N), _convert_to_sparse_format(H)


def _class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}


def _class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}


class BaseModel(object):
    """Base model for SVD and SVD++.
    """

    def __init__(self, config):
        self._built = False
        self._saver = None

        for attr in _class_vars(config):
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(config, attr))

    def save_model(self, model_dir):
        """Saves Tensorflow model.

        Args:
            model_dir: A string, the path of saving directory
        """

        if not self._built:
            raise RuntimeError('The model must be trained '
                               'before saving.')

        self._saver = tf.train.Saver()

        model_name = type(self).__name__

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, model_name)

        self._saver.save(self._sess, model_path)

    def load_model(self, model_dir):
        """Loads Tensorflow model.

        Args:
            model_dir: A string, the path of saving directory
        """

        tensor_names = ['placeholder/users:0', 'placeholder/items:0',
                        'placeholder/ratings:0', 'prediction/pred:0']
        operation_names = ['optimizer/optimizer']

        model_name = type(self).__name__

        model_path = os.path.join(model_dir, model_name)

        self._saver = tf.train.import_meta_graph(model_path + '.meta')
        self._saver.restore(self._sess, model_path)

        for name in tensor_names:
            attr = '_' + name.split('/')[1].split(':')[0]
            setattr(self, attr, tf.get_default_graph().get_tensor_by_name(name))

        for name in operation_names:
            attr = '_' + name.split('/')[1].split(':')[0]
            setattr(self, attr, tf.get_default_graph(
            ).get_operation_by_name(name))

        self._built = True


class SVD(BaseModel):
    """Collaborative filtering model based on SVD algorithm.
    """

    def __init__(self, config, sess):
        super(SVD, self).__init__(config)
        self._sess = sess

    def _create_placeholders(self):
        """Returns the placeholders.
        """
        with tf.variable_scope('placeholder'):
            users = tf.placeholder(tf.int32, shape=[None, ], name='users')
            items = tf.placeholder(tf.int32, shape=[None, ], name='items')
            ratings = tf.placeholder(
                tf.float32, shape=[None, ], name='ratings')

        return users, items, ratings

    def _create_constants(self, mu):
        """Returns the constants.
        """
        with tf.variable_scope('constant'):
            _mu = tf.constant(mu, shape=[], dtype=tf.float32)

        return _mu

    def _create_user_terms(self, users):
        """Returns the tensors related to users.
        """
        num_users = self.num_users
        num_factors = self.num_factors

        with tf.variable_scope('user'):
            user_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_users, num_factors],
                initializer=tff.initializers.GlorotUniform(),
                regularizer=tf.keras.regularizers.l2(self.reg_p_u))

            user_bias = tf.get_variable(
                name='bias',
                shape=[num_users, ],
                initializer=tff.initializers.GlorotUniform(),
                regularizer=tf.keras.regularizers.l2(self.reg_b_u))

            p_u = tf.nn.embedding_lookup(
                user_embeddings,
                users,
                name='p_u')

            b_u = tf.nn.embedding_lookup(
                user_bias,
                users,
                name='b_u')

        return p_u, b_u

    def _create_item_terms(self, items):
        """Returns the tensors related to items.
        """
        num_items = self.num_items
        num_factors = self.num_factors

        with tf.variable_scope('item'):
            item_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=tff.initializers.GlorotUniform(),
                regularizer=tf.keras.regularizers.l2(self.reg_q_i))

            item_bias = tf.get_variable(
                name='bias',
                shape=[num_items, ],
                initializer=tff.initializers.GlorotUniform(),
                regularizer=tf.keras.regularizers.l2(self.reg_b_i))

            q_i = tf.nn.embedding_lookup(
                item_embeddings,
                items,
                name='q_i')

            b_i = tf.nn.embedding_lookup(
                item_bias,
                items,
                name='b_i')

        return q_i, b_i

    def _create_prediction(self, mu, b_u, b_i, p_u, q_i):
        """Returns the tensor of prediction.

           Note that the prediction
            r_hat = \mu + b_u + b_i + p_u * q_i
        """
        with tf.variable_scope('prediction'):
            pred = tf.reduce_sum(
                tf.multiply(p_u, q_i),
                axis=1)

            pred = tf.add_n([b_u, b_i, pred])

            pred = tf.add(pred, mu, name='pred')

        return pred

    def _create_loss(self, pred, ratings):
        """Returns the L2 loss of the difference between
            ground truths and predictions.

           The formula is here:
            L2 = sum((r - r_hat) ** 2) / 2
        """
        with tf.variable_scope('loss'):
            loss = tf.nn.l2_loss(tf.subtract(ratings, pred), name='loss')

        return loss

    def _create_optimizer(self, loss):
        """Returns the optimizer.

           The objective function is defined as the sum of
            loss and regularizers' losses.
        """
        with tf.variable_scope('optimizer'):
            objective = tf.add(
                loss,
                tf.add_n(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)),
                name='objective')

            try:
                optimizer = tf.keras.optimizers.Nadam(
                ).minimize(objective, name='optimizer')
            except:
                optimizer = tf.train.AdamOptimizer().minimize(objective, name='optimizer')

        return optimizer

    def _build_graph(self, mu):
        _mu = self._create_constants(mu)

        self._users, self._items, self._ratings = self._create_placeholders()

        p_u, b_u = self._create_user_terms(self._users)
        q_i, b_i = self._create_item_terms(self._items)

        self._pred = self._create_prediction(_mu, b_u, b_i, p_u, q_i)

        loss = self._create_loss(self._ratings, self._pred)

        self._optimizer = self._create_optimizer(loss)

        self._built = True

    def _run_train(self, x, y, epochs, batch_size, validation_data):
        train_gen = BatchGenerator(x, y, batch_size)
        steps_per_epoch = np.ceil(train_gen.length / batch_size).astype(int)

        self._sess.run(tf.global_variables_initializer())

        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))

            pbar = utils.Progbar(steps_per_epoch)

            for step, batch in enumerate(train_gen.next(), 1):
                users = batch[0][:, 0]
                items = batch[0][:, 1]
                ratings = batch[1]

                self._sess.run(
                    self._optimizer,
                    feed_dict={
                        self._users: users,
                        self._items: items,
                        self._ratings: ratings
                    })

                pred = self.predict(batch[0])

                update_values = [
                    ('rmse', rmse(ratings, pred)),
                    ('mae', mae(ratings, pred))
                ]

                if validation_data is not None and step == steps_per_epoch:
                    valid_x, valid_y = validation_data
                    valid_pred = self.predict(valid_x)

                    update_values += [
                        ('val_rmse', rmse(valid_y, valid_pred)),
                        ('val_mae', mae(valid_y, valid_pred))
                    ]

                pbar.update(step, values=update_values)

    def train(self, x, y, epochs=100, batch_size=1024, validation_data=None):

        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        if not self._built:
            self._build_graph(np.mean(y))

        self._run_train(x, y, epochs, batch_size, validation_data)

    def predict(self, x):
        if not self._built:
            raise RuntimeError('The model must be trained '
                               'before prediction.')

        if x.shape[1] != 2:
            raise ValueError('The shape of x should be '
                             '(samples, 2)')

        pred = self._sess.run(
            self._pred,
            feed_dict={
                self._users: x[:, 0],
                self._items: x[:, 1]
            })

        pred = pred.clip(min=self.min_value, max=self.max_value)

        return pred


def _convert_to_sparse_format(x):
    """Converts a list of lists into sparse format.

    Args:
        x: A list of lists.

    Returns:
        A dictionary that contains three fields, which are
            indices, values, and the dense shape of sparse matrix.
    """

    sparse = {
        'indices': [],
        'values': []
    }

    for row, x_i in enumerate(x):
        for col, x_ij in enumerate(x_i):
            sparse['indices'].append((row, col))
            sparse['values'].append(x_ij)

    max_col = np.max([len(x_i) for x_i in x]).astype(np.int32)

    sparse['dense_shape'] = (len(x), max_col)

    return sparse


def _get_implicit_feedback(x, num_users, num_items, dual):
    """Gets implicit feedback from (users, items) pair.

    Args:
        x: A numpy array of shape `(samples, 2)`.
        num_users: An integer, total number of users.
        num_items: An integer, total number of items.
        dual: A bool, deciding whether returns the
            dual term of implicit feedback of items.

    Returns:
        A dictionary that is the sparse format of implicit
            feedback of users, if dual is true.
        A tuple of dictionarys that are the sparse format of
            implicit feedback of users and items, otherwise.
    """

    if not dual:
        N = [[] for u in range(num_users)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)

        return _convert_to_sparse_format(N)
    else:
        N = [[] for u in range(num_users)]
        H = [[] for u in range(num_items)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)
            H[i].append(u)

        return _convert_to_sparse_format(N), _convert_to_sparse_format(H)


class svdpp(SVD):
    """Collaborative filtering model based on SVD++ algorithm.
    """

    def __init__(self, config, sess, dual=False):
        super(svdpp, self).__init__(config, sess)
        self.dual = dual

    def _create_implicit_feedback(self, implicit_feedback, dual=False):
        """Returns the (tuple of) sparse tensor(s) of implicit feedback.
        """
        with tf.variable_scope('implicit_feedback'):
            if not dual:
                N = tf.SparseTensor(**implicit_feedback)

                return N
            else:
                N = tf.SparseTensor(**implicit_feedback[0])
                H = tf.SparseTensor(**implicit_feedback[1])

                return N, H

    def _create_user_terms(self, users, N):
        num_users = self.num_users
        num_items = self.num_items
        num_factors = self.num_factors

        p_u, b_u = super(svdpp, self)._create_user_terms(users)

        with tf.variable_scope('user'):
            implicit_feedback_embeddings = tf.get_variable(
                name='implict_feedback_embedding',
                shape=[num_items, num_factors],
                initializer=tf.zeros_initializer(),
                regularizer=tf.keras.regularizers.l2(self.reg_y_u))

            y_u = tf.gather(
                tf.nn.embedding_lookup_sparse(
                    implicit_feedback_embeddings,
                    N,
                    sp_weights=None,
                    combiner='sqrtn'),
                users,
                name='y_u'
            )

        return p_u, b_u, y_u

    def _create_item_terms(self, items, H=None):
        num_users = self.num_users
        num_items = self.num_items
        num_factors = self.num_factors

        q_i, b_i = super(svdpp, self)._create_item_terms(items)

        if H is None:
            return q_i, b_i
        else:
            with tf.variable_scope('item'):
                implicit_feedback_embeddings = tf.get_variable(
                    name='implict_feedback_embedding',
                    shape=[num_users, num_factors],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_g_i))

                g_i = tf.gather(
                    tf.nn.embedding_lookup_sparse(
                        implicit_feedback_embeddings,
                        H,
                        sp_weights=None,
                        combiner='sqrtn'),
                    items,
                    name='g_i'
                )

            return q_i, b_i, g_i

    def _create_prediction(self, mu, b_u, b_i, p_u, q_i, y_u, g_i=None):
        with tf.variable_scope('prediction'):
            if g_i is None:
                pred = tf.reduce_sum(
                    tf.multiply(tf.add(p_u, y_u), q_i),
                    axis=1)
            else:
                pred = tf.reduce_sum(
                    tf.multiply(tf.add(p_u, y_u), tf.add(q_i, g_i)),
                    axis=1)

            pred = tf.add_n([b_u, b_i, pred])

            pred = tf.add(pred, mu, name='pred')

        return pred

    def _build_graph(self, mu, implicit_feedback):
        _mu = super(svdpp, self)._create_constants(mu)

        self._users, self._items, self._ratings = super(
            svdpp, self)._create_placeholders()

        if not self.dual:
            N = self._create_implicit_feedback(implicit_feedback)

            p_u, b_u, y_u = self._create_user_terms(self._users, N)
            q_i, b_i = self._create_item_terms(self._items)

            self._pred = self._create_prediction(_mu, b_u, b_i, p_u, q_i, y_u)
        else:
            N, H = self._create_implicit_feedback(implicit_feedback, True)

            p_u, b_u, y_u = self._create_user_terms(self._users, N)
            q_i, b_i, g_i = self._create_item_terms(self._items, H)

            self._pred = self._create_prediction(
                _mu, b_u, b_i, p_u, q_i, y_u, g_i)

        loss = super(svdpp, self)._create_loss(self._ratings, self._pred)

        self._optimizer = super(svdpp, self)._create_optimizer(loss)

        self._built = True

    def train(self, x, y, epochs=100, batch_size=1024, validation_data=None):

        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        if not self._built:
            implicit_feedback = _get_implicit_feedback(
                x, self.num_users, self.num_items, self.dual)
            self._build_graph(np.mean(y), implicit_feedback)

        self._run_train(x, y, epochs, batch_size, validation_data)
