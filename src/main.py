import os
import sys
import yaml
import argparse
from src.train_model import train_model


def parse_args():
    """Parse arguments.

        Args:

        Returns:

        Raises:

    """
    parser = argparse.ArgumentParser(description='Model training tool.')
    parser.add_argument('--job', dest='job', default='job.yaml', help='Job description file')
    parser.add_argument('--eval-only', dest='eval_only', action='store_true', default=False, help='Run only evaluation')

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.job):
        print("File {} not found.".format(args.job))
        sys.exit(1)

    # Load YAML configuration
    config = yaml.load(open(args.job))

    # for name, conf in config.items():
    #     train_model(conf, eval_only=args.eval_only)


if __name__ == '__main__':
    main()