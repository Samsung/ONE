import argparse
import logging
import json
import typing

from lib.circle_plus import CirclePlus
from lib.train_info import TrainInfo


def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='circle plus generator',
        description='help handle circle file with training hyper parameters')

    parser.add_argument(
        'input_circle_file', metavar="input.circle", type=str, help='input circle file')

    args = parser.parse_args()
    return args


def check(in_circle_file) -> typing.NoReturn:
    '''
    Check in_circle_file has training hyperparameters and print it.
    '''
    circle_model: CirclePlus = CirclePlus.from_file(in_circle_file)
    tinfo = circle_model.get_train_info()

    print(f"check hyperparameters in {in_circle_file}")
    if tinfo == None:
        print("No hyperparameters")
    else:
        print(tinfo.dump_as_json())


if __name__ == "__main__":
    args = get_cmd_args()

    check(args.input_circle_file)

    # TODO: add a function that injects training parameter into circle file
