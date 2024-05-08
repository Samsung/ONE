import argparse
import logging
import json
from enum import Enum

from lib.circle_plus import CirclePlus
from lib.train_info import TrainInfo


def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='training hyperparameter injector',
        description='inject training parameter to the \'input_circle_file\'')
    parser.add_argument(
        'input_circle_file', metavar="input.circle", type=str, help='input circle file')
    parser.add_argument(
        'hyperparameter_json_file',
        type=str,
        nargs='?',
        metavar='hyper_param.json',
        help='input json file which has training hyper parameters')
    parser.add_argument(
        'output_circle_file',
        type=str,
        nargs='?',
        metavar="output.circle",
        help='output circle file with training parameter added\n'
        'if not given, input_circle_file is overwritten')
    parser.add_argument(
        '-v', '--verbosity', action="count", default=0, help='increase log verbosity')
    args = parser.parse_args()

    # if output_circle_file is not given, rewrite input_circle_file
    if not args.output_circle_file:
        args.output_circle_file = args.input_circle_file
    return args


def init_logger(verbosity: int):
    if verbosity >= 2:
        verbosity = 2
    log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=log_level[verbosity])


def check(in_file):
    circle_model = CirclePlus(in_file)
    tinfo = circle_model.get_tinfo()

    print(f"inspect hyperparameter in {in_file}")

    if tinfo == None:
        print("No hyper parameter")
    else:
        print(json.dumps(tinfo.dump(), indent=4))


def inject(in_file, param_file, out_file=None):
    if out_file == None:
        out_file = in_file

    with open(param_file, 'rt') as f:
        json_obj = json.load(f)

    logging.info(f"insert hyperparameter : ")
    logging.info(json.dumps(json_obj, indent=4))

    train_info = TrainInfo.from_json(json_obj)

    circle_model = CirclePlus(in_file)
    circle_model.inject_tinfo_as_metadata(train_info)
    circle_model.export(out_file)


if __name__ == "__main__":
    args = get_cmd_args()
    init_logger(args.verbosity)

    if args.hyperparameter_json_file is None:
        # if hyperparameter file isn't given,
        # just dump input circle file's hyperparameter
        check(args.input_circle_file)

    else:
        # inject hyperparamter_json to the input_circle_file
        inject(args.input_circle_file, args.hyperparameter_json_file,
               args.output_circle_file)
