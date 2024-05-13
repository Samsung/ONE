import argparse
import typing

from lib.circle_plus import CirclePlus
from lib.train_param import TrainParam


def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='circle plus generator',
        description='help handle circle file with training hyperparameters')

    parser.add_argument(
        'input_circle_file', metavar="input.circle", type=str, help='input circle file')

    args = parser.parse_args()
    return args


def print_training_hparameters(in_circle_file) -> typing.NoReturn:
    '''
    if in_circle_file has training hyperparameters, print it out
    '''
    print(f"check hyperparameters in {in_circle_file}")

    circle_model: CirclePlus = CirclePlus.from_file(in_circle_file)
    tinfo: typing.Union[TrainParam, None] = circle_model.get_train_param()

    if tinfo == None:
        print("No hyperparameters")
    else:
        print(tinfo.dump_as_json())


if __name__ == "__main__":
    args = get_cmd_args()

    print_training_hparameters(args.input_circle_file)

    # TODO: add a function that injects training parameter into circle file
