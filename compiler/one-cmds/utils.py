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
import importlib
import ntpath
import os
import subprocess
import sys

import python.constant as _constant


def _add_default_arg(parser):
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


def is_accumulated_arg(arg, driver):
    if driver == "one-quantize":
        accumulables = [
            "tensor_name", "scale", "zero_point", "src_tensor_name", "dst_tensor_name"
        ]
        if arg in accumulables:
            return True

    return False


def _is_valid_attr(args, attr):
    return hasattr(args, attr) and getattr(args, attr)


def _parse_cfg_and_overwrite(config_path, section, args):
    """
    parse given section of configuration file and set the values of args.
    Even if the values parsed from the configuration file already exist in args,
    the values are overwritten.
    """
    if config_path == None:
        # DO NOTHING
        return
    config = configparser.ConfigParser()
    # make option names case sensitive
    config.optionxform = str
    parsed = config.read(config_path)
    if not parsed:
        raise FileNotFoundError('Not found given configuration file')
    if not config.has_section(section):
        raise AssertionError('configuration file doesn\'t have \'' + section +
                             '\' section')
    for key in config[section]:
        setattr(args, key, config[section][key])
    # TODO support accumulated arguments


def _parse_cfg(args, driver_name):
    """parse configuration file. If the option is directly given to the command line,
       the option is processed prior to the configuration file.
       That is, if the values parsed from the configuration file already exist in args,
       the values are ignored."""
    if _is_valid_attr(args, 'config'):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(args.config)
        # if section is given, verify given section
        if _is_valid_attr(args, 'section'):
            if not config.has_section(args.section):
                raise AssertionError('configuration file must have \'' + driver_name +
                                     '\' section')
            for key in config[args.section]:
                if is_accumulated_arg(key, driver_name):
                    if not _is_valid_attr(args, key):
                        setattr(args, key, [config[args.section][key]])
                    else:
                        getattr(args, key).append(config[args.section][key])
                    continue
                if not _is_valid_attr(args, key):
                    setattr(args, key, config[args.section][key])
        # if section is not given, section name is same with its driver name
        else:
            if not config.has_section(driver_name):
                raise AssertionError('configuration file must have \'' + driver_name +
                                     '\' section')
            secton_to_run = driver_name
            for key in config[secton_to_run]:
                if is_accumulated_arg(key, driver_name):
                    if not _is_valid_attr(args, key):
                        setattr(args, key, [config[secton_to_run][key]])
                    else:
                        getattr(args, key).append(config[secton_to_run][key])
                    continue
                if not _is_valid_attr(args, key):
                    setattr(args, key, config[secton_to_run][key])


def _make_tf2tfliteV2_cmd(args, driver_path, input_path, output_path):
    """make a command for running tf2tfliteV2.py"""
    cmd = [sys.executable, os.path.expanduser(driver_path)]
    # verbose
    if _is_valid_attr(args, 'verbose'):
        cmd.append('--verbose')
    # model_format
    if _is_valid_attr(args, 'model_format_cmd'):
        cmd.append(getattr(args, 'model_format_cmd'))
    elif _is_valid_attr(args, 'model_format'):
        cmd.append('--' + getattr(args, 'model_format'))
    else:
        cmd.append('--graph_def')  # default value
    # converter version
    if _is_valid_attr(args, 'converter_version_cmd'):
        cmd.append(getattr(args, 'converter_version_cmd'))
    elif _is_valid_attr(args, 'converter_version'):
        cmd.append('--' + getattr(args, 'converter_version'))
    else:
        cmd.append('--v1')  # default value
    # input_path
    if _is_valid_attr(args, 'input_path'):
        cmd.append('--input_path')
        cmd.append(os.path.expanduser(input_path))
    # output_path
    if _is_valid_attr(args, 'output_path'):
        cmd.append('--output_path')
        cmd.append(os.path.expanduser(output_path))
    # input_arrays
    if _is_valid_attr(args, 'input_arrays'):
        cmd.append('--input_arrays')
        cmd.append(getattr(args, 'input_arrays'))
    # input_shapes
    if _is_valid_attr(args, 'input_shapes'):
        cmd.append('--input_shapes')
        cmd.append(getattr(args, 'input_shapes'))
    # output_arrays
    if _is_valid_attr(args, 'output_arrays'):
        cmd.append('--output_arrays')
        cmd.append(getattr(args, 'output_arrays'))

    return cmd


def _make_tflite2circle_cmd(driver_path, input_path, output_path):
    """make a command for running tflite2circle"""
    cmd = [driver_path, input_path, output_path]
    return [os.path.expanduser(c) for c in cmd]


def _make_circle2circle_cmd(args, driver_path, input_path, output_path):
    """make a command for running circle2circle"""
    cmd = [os.path.expanduser(c) for c in [driver_path, input_path, output_path]]
    # profiling
    if _is_valid_attr(args, 'generate_profile_data'):
        cmd.append('--generate_profile_data')
    # optimization pass(only true/false options)
    # TODO support options whose number of arguments is more than zero
    for opt in _constant.CONSTANT.OPTIMIZATION_OPTS:
        if _is_valid_attr(args, opt[0]):
            # ./driver --opt[0]
            if type(getattr(args, opt[0])) is bool:
                cmd.append('--' + opt[0])
            """
            This condition check is for config file interface, usually would be
             SomeOption=True
            but user can write as follows while development
             SomeOption=False
            instead of removing SomeOption option
            """
            if type(getattr(args, opt[0])) is str and not getattr(
                    args, opt[0]).lower() in ['false', '0', 'n']:
                cmd.append('--' + opt[0])

    return cmd


def _print_version_and_exit(file_path):
    """print version of the file located in the file_path"""
    script_path = os.path.realpath(file_path)
    dir_path = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    # run one-version
    subprocess.call([os.path.join(dir_path, 'one-version'), script_name])
    sys.exit()


def _safemain(main, mainpath):
    """execute given method and print with program name for all uncaught exceptions"""
    try:
        main()
    except Exception as e:
        prog_name = os.path.basename(mainpath)
        print(f"{prog_name}: {type(e).__name__}: " + str(e), file=sys.stderr)
        sys.exit(255)


def _run(cmd, err_prefix=None, logfile=None):
    """Execute command in subprocess

    Args:
        cmd: command to be executed in subprocess
        err_prefix: prefix to be put before every stderr lines
        logfile: file stream to which both of stdout and stderr lines will be written
    """
    with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1) as p:
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


def _remove_prefix(str, prefix):
    if str.startswith(prefix):
        return str[len(prefix):]
    return str


def _remove_suffix(str, suffix):
    if str.endswith(suffix):
        return str[:-len(suffix)]
    return str


def _get_optimization_list(get_name=False):
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
    files = [f for f in glob.glob(dir_path + '/../optimization/O*.cfg', recursive=True)]
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
        opt_list = [_remove_suffix(s, '.cfg') for s in opt_list]

    return opt_list


def _detect_one_import_drivers(search_path):
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
