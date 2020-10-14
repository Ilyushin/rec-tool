"""
Pipeline main function
"""

import os
import sys
import yaml
import argparse
import pandas as pd
from time import time
import tensorflow as tf
from datetime import datetime, timedelta
from cf_experiments_loop.common import fn
from cf_experiments_loop.train_model import train_model, train_svd, train_both_types
from cf_experiments_loop.ml_flow.ml_flow import log_to_mlflow, get_best_results_mlflow


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1:], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def parse_args():
    """Parse arguments.

        Args:

        Returns:

        Raises:

    """
    parser = argparse.ArgumentParser(description='Model training tool.')

    parser.add_argument("--model_dir", "-md", help="set path for the best model")
    parser.add_argument("--batch_size", "-b", help="set list of batch sizes for grid search")
    parser.add_argument("--epochs", "-ep", help="set number of epochs")

    parser.add_argument('--config',
                        dest='config',
                        default='config_example.yaml',
                        help='Configuration description file')

    parser.add_argument('--eval-only',
                        dest='eval_only',
                        action='store_true',
                        default=False,
                        help='Run only evaluation')

    return parser.parse_args()


def main():
    """
    :return:
    """

    # parse input arguments
    args = parse_args()

    if not os.path.exists(args.config):
        print("File {} not found.".format(args.config))
        sys.exit(1)

    # Load YAML configuration
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    train_data, test_data, users_number, items_number = None, None, None, None
    period, end_date = None, None
    dataset_name = None

    input_data_conf = config['config']['data']['input_data']

    if input_data_conf['movielens']['use']:
        transformations_fn = fn(input_data_conf['movielens']['transformations'])
        train_data, test_data, users_number, items_number = transformations_fn(
            dataset_type=input_data_conf['movielens']['type'],
            movielens_path=input_data_conf['movielens']['path']
        )
        dataset_name = config['config']['data']['input_data']['movielens']['type']

    if input_data_conf['goodreads']['use']:
        goodreads_transform = fn(input_data_conf['goodreads']['transformations'])
        train_data, test_data, users_number, items_number = goodreads_transform()
        dataset_name = config['config']['data']['input_data']['goodreads']['type']

    if input_data_conf['bookcrossing']['use']:
        bookcrossing_transform = fn(input_data_conf['bookcrossing']['transformations'])
        train_data, test_data, users_number, items_number = bookcrossing_transform()
        dataset_name = config['config']['data']['input_data']['bookcrossing']['type']

    if input_data_conf['behance']['use']:
        behance_transform = fn(input_data_conf['behance']['transformations'])
        train_data, test_data, users_number, items_number = behance_transform()
        dataset_name = config['config']['data']['input_data']['behance']['type']

    if users_number and items_number:
        model_conf = config['config']['model']
        model_fn = model_conf['model']
        loss_fn = fn(model_conf['loss'])
        metrics_fn = [fn(name) for name in model_conf['metrics']]

        if args.batch_size:
            batch_size = args.batch_size
        else:
            batch_size = model_conf['batch_size']

        if args.epochs:
            epoch = args.epoch
        else:
            epoch = model_conf['epoch']

        learning_rate = model_conf['learning_rate']
        grid_search = model_conf['grid_search']
        optimizers = model_conf['optimizers']
        result_conf = config['config']['result']

        if args.model_dir:
            model_dir = args.model_dir
        else:
            model_dir = result_conf['model']

        log_dir = result_conf['log']
        results_csv = result_conf['results_csv']
        clear = result_conf['clear']
        log_to_ml_flow = result_conf['log_to_ml_flow']

        df_results = pd.DataFrame()
        if os.path.exists(results_csv):
            df_results = pd.read_csv(results_csv)

        # define optimizers
        if optimizers == 'all':
            opts = [tf.keras.optimizers.SGD,
                    tf.keras.optimizers.RMSprop,
                    tf.keras.optimizers.Adam]

        else:
            opts = [tf.keras.optimizers.Adam]

        # Start grid search
        if grid_search and isinstance(batch_size, list) and isinstance(epoch, list):
            for model_path in model_fn:
                for batch in map(int, batch_size):
                    for ep in map(int, epoch):
                        for optimizer in opts:

                            start = time()

                            model, metrics = train_both_types(
                                model_path=model_path,
                                metric_names=model_conf['metrics'],
                                train_data=train_data,
                                test_data=test_data,
                                users_number=users_number,
                                items_number=items_number,
                                loss_fn=loss_fn,
                                metrics_fn=metrics_fn,
                                model_dir=model_dir,
                                log_dir=log_dir,
                                clear=clear,
                                batch_size=batch,
                                epoch=ep)

                            if log_to_ml_flow:

                                # write to MLFlow
                                log_to_mlflow(
                                    project_name='TrendMD experiments',
                                    group_name=fn(model_path).__name__,
                                    params={'batch_size': batch,
                                            'epoch': ep,
                                            'optimizer': 'Adam',
                                            'run_time': time() - start,
                                            'period': period,
                                            'end_date': end_date},
                                    metrics=metrics,
                                    tags={'dataset': dataset_name,
                                          'model_name': model_path},
                                    artifacts=[])

                                print('Uploaded to MLFlow')

                            # write to csv file
                            if df_results.empty:
                                df_results = pd.DataFrame(
                                    get_result(model_conf['metrics'],
                                               list(metrics.values()),
                                               model_path,
                                               batch,
                                               ep,
                                               optimizer,
                                               input_data_conf['movielens']['type'], as_list=True)
                                )
                            else:
                                df_results = df_results.append(
                                    get_result(model_conf['metrics'],
                                               list(metrics.values()),
                                               model_path,
                                               batch,
                                               ep,
                                               optimizer,
                                               input_data_conf['movielens']['type']),
                                    ignore_index=True)

                            df_results.to_csv(results_csv)
                            print('Save to csv')

        else:

            start = time()

            model_path = model_conf['model'][0]
            optimizer = opts[0]

            model, metrics = train_both_types(
                model_path=model_path,
                metric_names=model_conf['metrics'],
                train_data=train_data,
                test_data=test_data,
                users_number=users_number,
                items_number=items_number,
                loss_fn=loss_fn,
                metrics_fn=metrics_fn,
                model_dir=model_dir,
                log_dir=log_dir,
                clear=clear,
                batch_size=batch_size,
                epoch=epoch)

            if log_to_ml_flow:

                # write to MLFlow
                log_to_mlflow(
                    project_name='TrendMD experiments',
                    group_name=fn(model_path).__name__,
                    params={'batch_size': batch_size,
                            'epoch': epoch,
                            'optimizer': 'Adam',
                            'run_time': time() - start,
                            'period': period,
                            'end_date': end_date},
                    metrics=metrics,
                    tags={'dataset': dataset_name,
                          'model_name': model_path},
                    artifacts=[])

                print('Uploaded to MLFlow')

            # write to csv file
            if df_results.empty:
                df_results = pd.DataFrame(
                    get_result(model_conf['metrics'],
                               list(metrics.values()),
                               model_path,
                               batch_size,
                               epoch,
                               optimizer,
                               input_data_conf['movielens']['type'], as_list=True)
                )
            else:
                df_results = df_results.append(
                    get_result(model_conf['metrics'],
                               list(metrics.values()),
                               model_path,
                               batch_size,
                               epoch,
                               optimizer,
                               input_data_conf['movielens']['type']),
                    ignore_index=True)

            df_results.to_csv(results_csv)
            print('Save to csv')


def get_result(metrics, history_eval, model_path, batch_size, epoch, optimizer, dataset_name,
               as_list=False):
    """
    :param metrics:
    :param history_eval:
    :param model_path:
    :param batch_size:
    :param epoch:
    :param optimizer:
    :param dataset_name:
    :param as_list:
    :return:
    """
    result_dict = {
        fn(metric).__name__: ([history_eval[index + 1]] if as_list else history_eval[index + 1]) for
        index, metric in enumerate(metrics)
    }
    result_dict['loss'] = ([history_eval[0]] if as_list else history_eval[0])
    result_dict['model'] = [model_path] if as_list else model_path
    result_dict['batch_size'] = [batch_size] if as_list else batch_size
    result_dict['epoch'] = [epoch] if as_list else epoch
    result_dict['optimizer'] = [optimizer.__name__] if as_list else optimizer.__name__
    result_dict['dataset'] = [dataset_name] if as_list else dataset_name

    return result_dict


if __name__ == '__main__':
    main()
