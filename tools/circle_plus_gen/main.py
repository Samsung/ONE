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
    if circle_model has training parameters, print it out
    '''
    tinfo: typing.Union[TrainParam, None] = circle_model.get_train_param()

    if tinfo == None:
        print("No hyperparameters")
    else:
        print(tinfo.dump_as_json())
        print()

        print("trainable operators :")
        trainable = tinfo.get_trainable()
        if len(trainable) == 0:
            print("\tno trainable operators")
        else:
            operators = circle_model.get_operators(subgraph_idx=0)
            for i in tinfo.get_trainable():
                print(f"\t[{i}] {operators[i]}")


if __name__ == "__main__":
    args = get_cmd_args()

    circle_model: CirclePlus = CirclePlus.from_file(args.input)

    if args.hyperparameters is not None:
        tparams = TrainParam.from_json(circle_model, args.hyperparameters)
        circle_model.set_train_param(tparams)

    print_training_hparameters(circle_model)

    # if out_file isn't given, rewrite in_file
    out_file = args.output
    if out_file is None:
        out_file = args.input

    circle_model.export(out_file)
