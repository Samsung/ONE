'''Parse arguments'''

import argparse


def _create_parser():
    parser = argparse.ArgumentParser(
        description='Convert a dataset of tensorflow to onert format',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-s', '--show-datasets', action='store_true', help='show dataset list')
    parser.add_argument(
        '-d',
        '--dataset-name',
        type=str,
        default='fashion_mnist',
        metavar='Dataset',
        help='name of dataset to be converted (default: "fashion_mnist")')
    parser.add_argument(
        '-o',
        '--out-dir',
        type=str,
        default='out',
        metavar='Dir',
        help='relative path of the files to be created (default: "out")')
    parser.add_argument(
        '-p',
        '--prefix-name',
        type=str,
        default='',
        metavar='Prefix',
        help='prefix name of the file to be created (default: "")')
    parser.add_argument(
        '--split',
        nargs='*',
        type=str,
        default=['train', 'test'],
        metavar='Split',
        help='Which split of the data to load (default: "train test")')
    parser.add_argument(
        '--length',
        nargs='*',
        type=int,
        default=[1000, 100],
        metavar='N',
        help='Data number for items described in split (default: "1000 100")')
    models = ['mnist', 'mobilenetv2']
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='mnist',
        choices=models,
        metavar='Model',
        help=('Model name to use generated data (default: mnist)\n'
              'Supported models: ' + ', '.join(models)))

    return parser


def parse_args():
    parser = _create_parser()
    args = parser.parse_args()

    return args
