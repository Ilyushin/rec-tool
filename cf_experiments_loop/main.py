import os
import sys
import yaml
import argparse
from cf_experiments_loop.common import fn
from cf_experiments_loop.train_model import train_model


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

    print(users_number, items_number)
    if users_number and items_number:
        model_conf = config['config']['model']
        model_fn = fn(model_conf['model'])
        loss_fn = fn(model_conf['loss'])
        metrics_fn = [fn(name) for name in model_conf['metrics']]
        batch_size = int(model_conf['batch_size'])
        epoch = int(model_conf['epoch'])

        result_conf = config['config']['result']
        model_dir = result_conf['model']
        log_dir = result_conf['log']
        clear = result_conf['clear']

        train_model(
            train_data=train_data,
            test_data=test_data,
            users_number=users_number,
            items_number=users_number,
            model_fn=model_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            batch_size=batch_size,
            epoch=epoch,
            model_dir=model_dir,
            log_dir=log_dir,
            clear=clear
        )


if __name__ == '__main__':
    main()
