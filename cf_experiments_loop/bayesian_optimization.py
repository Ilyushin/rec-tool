import pandas as pd
from bayes_opt import BayesianOptimization


def fit_bayesian_opt(batch_size: list, epoch: list, partial_train_model):
    """
    :param batch_size:
    :param epoch:
    :return:
    """

    pbounds = {'batch_size': tuple(batch_size), 'epochs': tuple(epoch)}

    optimizer = BayesianOptimization(
        f=partial_train_model,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=10)

    opt_results = pd.DataFrame({''.join(['step ', str(step)]): res for step, res in enumerate(optimizer.res)})

    return optimizer.max, opt_results
