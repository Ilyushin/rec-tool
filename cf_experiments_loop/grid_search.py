from sklearn.model_selection import GridSearchCV
import pandas as pd


def fit_grid_search(batch_size: list, epoch: list, model, features, targets):
    """
    :param batch_size:
    :param epoch:
    :param model:
    :param features:
    :param targets:
    :return:
    """

    # TODO: add optimizers
    param_grid = dict(batch_size=batch_size, epochs=epoch)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(features, targets)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    gs_results = [(mean, stdev, param) for mean, stdev, param in zip(means, stds, params)]
    return grid_result.best_score_, grid_result.best_params_,\
           pd.DataFrame({''.join(['step ', str(step)]): result for step, result in enumerate(gs_results)})


