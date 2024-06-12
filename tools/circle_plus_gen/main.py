import argparse
import typing

from lib.circle_plus import CirclePlus
from lib.train_param import TrainParam


def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='circle_plus_gen',
        description='circle_plus_gen help handle circle file with training hyperparameters'
    )

    parser.add_argument('input', help='input circle file')
    parser.add_argument(
        'hyperparameters', nargs='?', help='training hyperparameters json file')
    parser.add_argument(
        'output',
        nargs='?',
        help='output circle file, if not given input circle file will be overwritten')

    args = parser.parse_args()
    return args


def print_training_hparameters(circle_model: CirclePlus):
    '''
    if in_circle_model has training hyperparameters, print it out
    '''
    tinfo: typing.Union[TrainParam, None] = circle_model.get_train_param()

    if tinfo == None:
        print("No hyperparameters")
    else:
        print(tinfo.dump_as_json())
        # TODO print list of trainable operators


def inject_hparams(in_file, hparams_file, out_file=None) -> None:
    '''
    Inject hparams_file's contents into in_file's circle model, and save it as out_file 
    '''
    # if out_file isn't given, rewrite in_file
    if out_file is None:
        out_file = in_file

    tparams: TrainParam = TrainParam.from_json(hparams_file)
    circle_model: CirclePlus = CirclePlus.from_file(in_file)
    circle_model.set_train_param(tparams)
    print("succesfully add hyperparameters to the circle file")

    print_training_hparameters(circle_model)

    circle_model.export(out_file)
    print(f"saved in {out_file}")


if __name__ == "__main__":
    args = get_cmd_args()

    if args.hyperparameters is None:
        circle_model = CirclePlus.from_file(args.input)
        print_training_hparameters(circle_model)

    else:
        inject_hparams(args.input, args.hyperparameters, args.output)
