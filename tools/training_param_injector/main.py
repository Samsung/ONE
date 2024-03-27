import argparse
import logging
import json

import lib.json_parser as jparser
import lib.circle_builder as cbuilder
import lib.train_info as tinfo


def get_cmd_args():
    parser = argparse.ArgumentParser(
        prog='circle_parameter_injector',
        description='inject training parameter to the \'input_circle_file\'')
    parser.add_argument('input_circle_file', type=str, help='input circle file')
    parser.add_argument(
        'output_circle_file',
        type=str,
        nargs='?',
        help='output circle file with training parameter added\n'
        'if not given, input_circle_file is overwritten')
    parser.add_argument(
        '-t',
        '--train_param',
        type=str,
        metavar='abc.json',
        help='input json file which has training parameters')
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


def get_train_info(json_file=''):
    if json_file:
        logging.debug("json loaded")
        train_param = jparser.load(json_file)
    else:
        train_param = tinfo.TrainingInfo.create_from_input()
    return train_param


if __name__ == "__main__":
    args = get_cmd_args()
    init_logger(args.verbosity)

    train_info = get_train_info(args.train_param)
    train_info_str = json.dumps(train_info, indent=4, cls=tinfo.JEcoder)
    logging.debug(train_info_str)

    circle_model = cbuilder.CircleModel(args.input_circle_file)
    circle_model.inject_tinfo_as_metadata(train_info)
    circle_model.export(args.output_circle_file)
