'''Parse arguments'''

import argparse


def _create_parser():
    parser = argparse.ArgumentParser(
        description='Convert a dataset of tensorflow to onert format')
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
        '-l',
        '--train-length',
        type=int,
        default=1000,
        metavar='N',
        help='Number of data for training (default: 1000)')
    parser.add_argument(
        '-t',
        '--test-length',
        type=int,
        default=100,
        metavar='N',
        help='Number of data for test (default: 100)')

    return parser


def parse_args():
    parser = _create_parser()
    args = parser.parse_args()

    return args
