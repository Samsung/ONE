#!/usr/bin/env python

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import glob
import importlib.machinery
import importlib.util
import ntpath
import os
import subprocess
import sys

from typing import Union

import onelib.constant as _constant


def add_default_arg(parser):
    # version
    parser.add_argument(
        '-v',
        '--version',
        action='store_true',
        help='show program\'s version number and exit')

    # verbose
    parser.add_argument(
        '-V',
        '--verbose',
        action='store_true',
        help='output additional information to stdout or stderr')

    # configuration file
    parser.add_argument('-C', '--config', type=str, help='run with configuation file')
    # section name that you want to run in configuration file
    parser.add_argument('-S', '--section', type=str, help=argparse.SUPPRESS)


def add_default_arg_no_CS(parser):
    """
    This adds -v -V args only (no -C nor -S)
    """
    # version
    parser.add_argument(
        '-v',
        '--version',
        action='store_true',
        help='show program\'s version number and exit')

    # verbose
    parser.add_argument(
        '-V',
        '--verbose',
        action='store_true',
        help='output additional information to stdout or stderr')


def is_accumulated_arg(arg, driver):
    if driver == "one-quantize":
        accumulables = [
            "tensor_name", "scale", "zero_point", "src_tensor_name", "dst_tensor_name"
        ]
        if arg in accumulables:
            return True

    return False


def is_valid_attr(args, attr):
    return hasattr(args, attr) and getattr(args, attr)


def get_config_parser() -> configparser.ConfigParser:
    """
    Initialize configparser and set default option

    This funciton has been introduced for all the one-cmds tools having same parsing option.
    """
    parser = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    parser.optionxform = str

    return parser


def parse_cfg(config_path: Union[str, None], section_to_parse: str, args):
    """
    parse configuration file and store the information to args
    
    :param config_path: path to configuration file
    :param section_to_parse: section name to parse
    :param args: object to store the parsed information
    """
    if config_path is None:
        return

    parser = get_config_parser()
    parser.read(config_path)

    if not parser.has_section(section_to_parse):
        raise AssertionError('configuration file must have \'' + section_to_parse +
                             '\' section')

    for key in parser[section_to_parse]:
        if is_accumulated_arg(key, section_to_parse):
            if not is_valid_attr(args, key):
                setattr(args, key, [parser[section_to_parse][key]])
            else:
                getattr(args, key).append(parser[section_to_parse][key])
            continue
        if hasattr(args, key) and getattr(args, key):
            continue
        setattr(args, key, parser[section_to_parse][key])


def print_version_and_exit(file_path):
    """print version of the file located in the file_path"""
    script_path = os.path.realpath(file_path)
    dir_path = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    # run one-version
    subprocess.call([os.path.join(dir_path, 'one-version'), script_name])
    sys.exit()


def safemain(main, mainpath):
    """execute given method and print with program name for all uncaught exceptions"""
    try:
        main()
    except Exception as e:
        prog_name = os.path.basename(mainpath)
        print(f"{prog_name}: {type(e).__name__}: " + str(e), file=sys.stderr)
        sys.exit(255)


def run(cmd, err_prefix=None, logfile=None):
    """Execute command in subprocess

    Args:
        cmd: command to be executed in subprocess
        err_prefix: prefix to be put before every stderr lines
        logfile: file stream to which both of stdout and stderr lines will be written
    """
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        import select
        inputs = set([p.stdout, p.stderr])
        while inputs:
            readable, _, _ = select.select(inputs, [], [])
            for x in readable:
                line = x.readline()
                if len(line) == 0:
                    inputs.discard(x)
                    continue
                if x == p.stdout:
                    out = sys.stdout
                if x == p.stderr:
                    out = sys.stderr
                    if err_prefix:
                        line = f"{err_prefix}: ".encode() + line
                out.buffer.write(line)
                out.buffer.flush()
                if logfile != None:
                    logfile.write(line)
    if p.returncode != 0:
        sys.exit(p.returncode)


def remove_prefix(str, prefix):
    if str.startswith(prefix):
        return str[len(prefix):]
    return str


def remove_suffix(str, suffix):
    if str.endswith(suffix):
        return str[:-len(suffix)]
    return str


def get_optimization_list(get_name=False):
    """
    returns a list of optimization. If `get_name` is True,
    only basename without extension is returned rather than full file path.

    [one hierarchy]
    one
    ├── backends
    ├── bin
    ├── doc
    ├── include
    ├── lib
    ├── optimization
    └── test

    Optimization options must be placed in `optimization` folder
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # optimization folder
    files = [
        f for f in glob.glob(dir_path + '/../../optimization/O*.cfg', recursive=True)
    ]
    # exclude if the name has space
    files = [s for s in files if not ' ' in s]

    opt_list = []
    for cand in files:
        base = ntpath.basename(cand)
        if os.path.isfile(cand) and os.access(cand, os.R_OK):
            opt_list.append(cand)

    if get_name == True:
        # NOTE the name includes prefix 'O'
        # e.g. O1, O2, ONCHW not just 1, 2, NCHW
        opt_list = [ntpath.basename(f) for f in opt_list]
        opt_list = [remove_suffix(s, '.cfg') for s in opt_list]

    return opt_list


def detect_one_import_drivers(search_path):
    """Looks for import drivers in given directory

    Args:
        search_path: path to the directory where to search import drivers

    Returns:
    dict: each entry is related to single detected driver,
          key is a config section name, value is a driver name

    """
    import_drivers_dict = {}
    for module_name in os.listdir(search_path):
        full_path = os.path.join(search_path, module_name)
        if not os.path.isfile(full_path):
            continue
        if module_name.find("one-import-") != 0:
            continue
        module_loader = importlib.machinery.SourceFileLoader(module_name, full_path)
        module_spec = importlib.util.spec_from_loader(module_name, module_loader)
        module = importlib.util.module_from_spec(module_spec)
        try:
            module_loader.exec_module(module)
            if hasattr(module, "get_driver_cfg_section"):
                section = module.get_driver_cfg_section()
                import_drivers_dict[section] = module_name
        except:
            pass
    return import_drivers_dict
