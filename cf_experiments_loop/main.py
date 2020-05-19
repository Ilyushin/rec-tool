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
        model_fn = [fn(model) for model in model_conf['model']]
        loss_fn = fn(model_conf['loss'])
        metrics_fn = [fn(name) for name in model_conf['metrics']]
        batch_size = model_conf['batch_size']
        epoch = model_conf['epoch']
        grid_search = model_conf['grid_search']
        optimizers = model_conf['optimizers']
        result_conf = config['config']['result']
        model_dir = result_conf['model']
        log_dir = result_conf['log']
        results_csv = result_conf['results_csv']
        clear = result_conf['clear']
        log_to_ml_flow = result_conf['log_to_ml_flow']

        # define optimizers
        if optimizers == 'all':
            opts = [tf.keras.optimizers.SGD(),
                    tf.keras.optimizers.RMSprop(),
                    tf.keras.optimizers.Adam()]

        else:
            opts = [tf.keras.optimizers.Adam()]

        # Start grid search
        if grid_search and isinstance(batch_size, list) and isinstance(epoch, list):
            for model in model_fn:
                for batch in map(int, batch_size):
                    for e in map(int, epoch):
                        for optimizer in opts:

                            start = time()

                            history_train, history_eval = train_model(
                                train_data=train_data,
                                test_data=test_data,
                                users_number=users_number,
                                items_number=users_number,
                                model_fn=model,
                                loss_fn=loss_fn,
                                metrics_fn=metrics_fn,
                                model_dir=model_dir,
                                log_dir=log_dir,
                                clear=clear,
                                batch_size=batch,
                                epoch=e,
                                optimizer=optimizer
                            )
                            print(history_eval[-1])

                            # write to MLFlow
                            if log_to_ml_flow:
                                log_to_mlflow(project_name='Recommendation system results',
                                              group_name=str(model),
                                              params={'batch_size': batch_size,
                                                      'epoch': epoch,
                                                      'optimizer': str(optimizer),
                                                      'run_time': time() - start},
                                              metrics={'eval': history_eval[-1], 'time': time() - start},
                                              tags={'dataset': 'movielens'},
                                              artifacts=[model_dir])

                                print('uploaded to MLFlow')
                                print(history_eval[0])

                            # write to csv file
                            pd.DataFrame({
                                'model': str(model),
                                'batch_size': batch_size,
                                'epoch': epoch,
                                'optimizer': str(optimizer),
                                'results': history_eval[0]
                            }).to_csv('results.csv')

        else:

            start = time()

            history_train, history_eval = train_model(
                train_data=train_data,
                test_data=test_data,
                users_number=users_number,
                items_number=users_number,
                model_fn=model_fn,
                loss_fn=loss_fn,
                metrics_fn=metrics_fn,
                model_dir=model_dir,
                log_dir=log_dir,
                clear=clear,
                batch_size=batch_size,
                epoch=epoch,
                optimizer=tf.keras.optimizers.Adam()
            )

            # write to MLFlow
            if log_to_ml_flow:
                log_to_mlflow(project_name='Recommendations',
                              group_name=str(model_fn),
                              params={'batch_size': batch_size,
                                      'epoch': epoch,
                                      'optimizer': 'Adam',
                                      'run_time': time() - start},
                              metrics={'metrics': history_eval[-1]},
                              tags={'dataset': 'movielens'},
                              artifacts=[model_dir, results_csv])

            # write to csv file
            pd.DataFrame({
                'model': str(model_fn),
                'batch_size': batch_size,
                'epoch': epoch,
                'optimizer': 'Adam',
                'results': history_eval[-1]
            }).to_csv(results_csv)


if __name__ == '__main__':
    main()
