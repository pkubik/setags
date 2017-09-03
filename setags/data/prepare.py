import argparse
from pathlib import Path

import setags.data.preprocessing as preprocessing
import setags.data.utils as du


DATA_FILENAMES = ['biology.csv', 'cooking.csv', 'diy.csv', 'robotics.csv', 'travel.csv']


class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Prepare data for the model.')
        parser.add_argument('--data_dir', '-d', metavar='DATA_DIR', default=None, type=str, help='data directory')

        args = parser.parse_args()

        self.data_dir = args.data_dir


def main():
    cli = CLI()
    if cli.data_dir is None:
        data_dir = du.get_data_dir()
    else:
        data_dir = Path(cli.data_dir)
    data_filenames = DATA_FILENAMES
    print("Using '{}' as data directory.".format(str(data_dir)))
    print("Preprocessing files {}.".format(data_filenames))
    preprocessing.prepare_data(data_filenames, data_dir, train_fraction=0.9)


if __name__ == '__main__':
    main()
