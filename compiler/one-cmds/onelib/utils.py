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

from typing import Union, Optional

from onelib.argumentparse import ArgumentParser
"""
Support commands in the one-cmds.

{ 
  ${GROUP} : {
    ${CMD} : EXPLANATIONS
  }
}
"""
ONE_CMD = {
    'compile': {
        'import': 'Convert given model to circle',
        'optimize': 'Optimize circle model',
        'quantize': 'Quantize circle model',
    },
    'package': {
        'pack': 'Package circle and metadata into nnpackage',
    },
    'backend': {
        'codegen': 'Code generation tool',
        'profile': 'Profile backend model file',
        'infer': 'Infer backend model file'
    },
}


def one_cmd_list():
    return [cmd for group, cmds in ONE_CMD.items() for cmd in cmds.keys()]


def add_default_arg(parser):
    # version
    parser.add_argument('-v',
                        '--version',
                        action='store_true',
                        help='show program\'s version number and exit')

    # verbose
    parser.add_argument('-V',
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
    parser.add_argument('-v',
                        '--version',
                        action='store_true',
                        help='show program\'s version number and exit')

    # verbose
    parser.add_argument('-V',
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


def parse_cfg(config_path: Union[str, None],
              section_to_parse: str,
              args,
              quiet: bool = False):
    """
    parse configuration file and store the information to args
    
    :param config_path: path to configuration file
    :param section_to_parse: section name to parse
    :param args: object to store the parsed information
    :param quiet: raise no error when given section doesn't exist
    """
    if config_path is None:
        return

    parser = get_config_parser()
    parser.read(config_path)

    if not parser.has_section(section_to_parse) and quiet:
        return

    if not parser.has_section(section_to_parse):
        raise AssertionError('configuration file must have \'' + section_to_parse +
                             '\' section')

    # set environment
    CFG_ENV_SECTION = 'Environment'
    if parser.has_section(CFG_ENV_SECTION):
        for key in parser[CFG_ENV_SECTION]:
            os.environ[key] = parser[CFG_ENV_SECTION][key]

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


def run_ret(cmd, *, one_cmd: str = None, err_prefix=None, logfile=None):
    """Execute command in subprocess

    Args:
        one_cmd: subtool name to execute with given `cmd`
        cmd: command to be executed in subprocess
        err_prefix: prefix to be put before every stderr lines
        logfile: file stream to which both of stdout and stderr lines will be written
    Return:
        Process execution return code; 0 if success and others for error.
    """
    if one_cmd:
        assert one_cmd in one_cmd_list(), f'Invalid ONE COMMAND: {one_cmd}'
        dir_path = os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))  # bin = onelib/../
        driver_path = os.path.join(dir_path, f'one-{one_cmd}')
        cmd = [driver_path] + cmd

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
    return p.returncode


# TODO make run call run_ret
def run(cmd, *, one_cmd: str = None, err_prefix=None, logfile=None):
    """Execute command in subprocess

    Args:
        one_cmd: subtool name to execute with given `cmd`
        cmd: command to be executed in subprocess
        err_prefix: prefix to be put before every stderr lines
        logfile: file stream to which both of stdout and stderr lines will be written
    """
    if one_cmd:
        assert one_cmd in one_cmd_list(), f'Invalid ONE COMMAND: {one_cmd}'
        dir_path = os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))  # bin = onelib/../
        driver_path = os.path.join(dir_path, f'one-{one_cmd}')
        cmd = [driver_path] + cmd

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


def get_target_list(get_name=False):
    """
    returns a list of targets. If `get_name` is True,
    only basename without extension is returned rather than full file path.

    [one hierarchy]
    one
    ├── backends
    ├── bin
    ├── doc
    ├── include
    ├── lib
    ├── optimization
    ├── target
    └── test

    Target configuration files must be placed in `target` folder
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # target folder
    files = [f for f in glob.glob(dir_path + '/../../target/*.ini', recursive=True)]
    # exclude if the name has space
    files = [s for s in files if not ' ' in s]

    target_list = []
    for cand in files:
        if os.path.isfile(cand) and os.access(cand, os.R_OK):
            target_list.append(cand)

    if get_name == True:
        target_list = [ntpath.basename(f) for f in target_list]
        target_list = [remove_suffix(s, '.ini') for s in target_list]

    return target_list


def get_arg_parser(backend: Optional[str], cmd: str,
                   target: Optional[str]) -> Optional[ArgumentParser]:
    if not backend:
        return None

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # for python module naming convention
    command_schema_path = dir_path + f'/../../backends/command/{backend}/{cmd}.py'
    if not os.path.isfile(command_schema_path):
        return None

    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(cmd, command_schema_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[cmd] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "command_schema"):
        raise RuntimeError('You must implement "command_schema" function')

    parser: ArgumentParser = module.command_schema()
    parser.target = target
    return parser


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
