import os
import sys
import yaml
import argparse
import pandas as pd
from time import time
import tensorflow as tf
from cf_experiments_loop.common import fn
from cf_experiments_loop.train_model import train_model
from cf_experiments_loop.ml_flow.ml_flow import log_to_mlflow


def parse_args():
    """Parse arguments.

        Args:

        Returns:

        Raises:

    """
    parser = argparse.ArgumentParser(description='Model training tool.')
    parser.add_argument('--config', dest='config', default='config_example.yaml', help='Configuration description file')
    parser.add_argument('--eval-only', dest='eval_only', action='store_true', default=False, help='Run only evaluation')

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        print("File {} not found.".format(args.config))
        sys.exit(1)

    # Load YAML configuration
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    train_data, test_data, users_number, items_number = None, None, None, None

    input_data_conf = config['config']['data']['input_data']
    if input_data_conf['movielens']['use']:
        transformations_fn = fn(input_data_conf['transformations'])
        train_data, test_data, users_number, items_number = transformations_fn(
            dataset_type=input_data_conf['movielens']['type'],
            clear=input_data_conf['clear'],
            movielens_path=input_data_conf['movielens']['path']
        )

    if users_number and items_number:
        model_conf = config['config']['model']
        dataset_name = config['config']['data']['input_data']['movielens']['type']
        model_fn = model_conf['model']
        loss_fn = fn(model_conf['loss'])
        metrics_fn = [fn(name) for name in model_conf['metrics']]
        batch_size = model_conf['batch_size']
        epoch = model_conf['epoch']
        # learning_rate = model_conf['learning_rate']
        grid_search = model_conf['grid_search']
        optimizers = model_conf['optimizers']
        result_conf = config['config']['result']
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
                    for e in map(int, epoch):
                        for optimizer in opts:

                            start = time()

                            history_train, history_eval = train_model(
                                train_data=train_data,
                                test_data=test_data,
                                users_number=users_number,
                                items_number=users_number,
                                model_fn=fn(model_path),
                                loss_fn=loss_fn,
                                metrics_fn=metrics_fn,
                                model_dir=model_dir,
                                log_dir=log_dir,
                                clear=clear,
                                batch_size=batch,
                                epoch=e,
                                optimizer=optimizer()
                            )
                            print('history_eval:', history_eval)

                            if log_to_ml_flow:
                                # write to MLFlow
                                log_to_mlflow(project_name='Recommendation system experiments',
                                              group_name=fn(model_path).__name__,
                                              params={'batch_size': batch,
                                                      'epoch': e,
                                                      'optimizer': 'Adam',
                                                      'run_time': time() - start},
                                              metrics={
                                                  metric.split('.')[-1]:
                                                      history_eval[model_conf['metrics'].index(metric) + 1]
                                                  for metric in model_conf['metrics']
                                              },
                                              tags={'dataset': dataset_name},
                                              artifacts=[model_dir])

                                print('Uploaded to MLFlow')

                            # write to csv file
                            if df_results.empty:
                                df_results = pd.DataFrame(
                                    get_result(model_conf['metrics'], history_eval, model_path, batch,
                                               e, optimizer,
                                               input_data_conf['movielens']['type'], as_list=True)
                                )
                            else:
                                df_results = df_results.append(
                                    get_result(model_conf['metrics'], history_eval, model_path, batch,
                                               e, optimizer, input_data_conf['movielens']['type']),
                                    ignore_index=True)

                            df_results.to_csv(results_csv)
                            print('Save to csv')

        else:

            start = time()

            model_path = model_conf['model'][0]
            optimizer = opts[0]
            history_train, history_eval = train_model(
                train_data=train_data,
                test_data=test_data,
                users_number=users_number,
                items_number=users_number,
                model_fn=fn(model_path),
                loss_fn=loss_fn,
                metrics_fn=metrics_fn,
                model_dir=model_dir,
                log_dir=log_dir,
                clear=clear,
                batch_size=batch_size,
                epoch=epoch,
                optimizer=optimizer()
            )

            # write to MLFlow
            if log_to_ml_flow:

                # write to MLFlow
                log_to_mlflow(project_name='Recommendation system experiments',
                              group_name=model_fn.__name__,
                              params={'batch_size': batch_size,
                                      'epoch': epoch,
                                      'optimizer': 'Adam',
                                      'run_time': time() - start},
                              metrics={
                                  metric.split('.')[-1]:
                                      history_eval[model_conf['metrics'].index(metric) + 1]
                                  for metric in model_conf['metrics']
                              },
                              tags={'dataset': dataset_name},
                              artifacts=[model_dir])

            # write to csv file
            if df_results.empty:
                df_results = pd.DataFrame(
                    get_result(model_conf['metrics'], history_eval, model_path, batch_size, epoch, optimizer,
                               [input_data_conf['movielens']['type']], as_list=True)
                )
            else:
                df_results = df_results.append(
                    get_result(model_conf['metrics'], history_eval, model_path, batch_size, epoch, optimizer,
                               [input_data_conf['movielens']['type']]),
                    ignore_index=True)

            df_results.to_csv(results_csv)
            print('Save to csv')


def get_result(metrics, history_eval, model_path, batch_size, epoch, optimizer, dataset_name,
               as_list=False):
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
