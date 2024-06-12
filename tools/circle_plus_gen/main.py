import argparse
import typing

from lib.circle_plus import CirclePlus
from lib.train_param import TrainParam


def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='circle_plus_gen',
        description=
        'circle_plus_gen help handle circle file with training hyperparameters',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('input', help='input circle file')
    parser.add_argument(
        'hyperparameters', nargs='?', help='training hyperparameters json file')
    parser.add_argument(
        '--output',
        nargs='?',
        help='output circle file, if not given input circle file will be overwritten')
    parser.add_argument(
        '--train',
        help='select which operator will be trained\n'
        '[all, none, lastN(e.g. last2, last3..)] are possible')
    args = parser.parse_args()
    return args


def print_training_hparameters(circle_model: CirclePlus):
    '''
    if circle_model has training parameters, print it out
    '''
    tinfo: typing.Union[TrainParam, None] = circle_model.get_train_param()
    operators = circle_model.get_operators(subgraph_idx=0)

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
            for i in tinfo.get_trainable():
                print(f"\t[{i}] {operators[i]}")


def to_trainable_index(s: str, num_op: int) -> typing.List[int]:
    s = s.lower()

    if s == "all":
        return list(range(0, num_op))
    elif s == "none":
        return []
    elif s.startswith("last"):
        try:
            last_n = int(s[4:])  #remvoe 'last'
        except ValueError as e:
            print(e)
            raise ValueError(f"lastN: N is not integer value")

        if last_n < 0:
            raise ValueError(f"lastN : N is negative value")

        if last_n > num_op:
            raise ValueError(
                f"number of operators({num_op}) < number of trainable operators({last_n})"
            )

        start_idx = num_op - last_n
        return list(range(start_idx, num_op))
    else:
        raise RuntimeError(f"not supported train {s}")


if __name__ == "__main__":
    args = get_cmd_args()

    circle_model: CirclePlus = CirclePlus.from_file(args.input)

    if args.hyperparameters is not None:
        tparams = TrainParam.from_json(args.hyperparameters)
        circle_model.set_train_param(tparams)

    if args.train is not None:
        tparam = circle_model.get_train_param()
        if tparam is None:
            tparam = TrainParam()

        num_operators = len(circle_model.get_operators())
        indexes = to_trainable_index(args.train, num_operators)
        tparam.set_trainable(indexes)
        circle_model.set_train_param(tparam)

    print_training_hparameters(circle_model)

    # if out_file isn't given, rewrite in_file
    out_file = args.output
    if out_file is None:
        out_file = args.input

    circle_model.export(out_file)
