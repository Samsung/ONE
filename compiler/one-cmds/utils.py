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
import os
import subprocess
import sys


def _add_default_arg(parser):
    # version
    parser.add_argument(
        '-v',
        '--version',
        action='store_true',
        help='show program\'s version number and exit')

    # configuration file
    parser.add_argument('-C', '--config', type=str, help='run with configuation file')
    # section name that you want to run in configuration file
    parser.add_argument('-S', '--section', type=str, help=argparse.SUPPRESS)


def _is_valid_attr(args, attr):
    return hasattr(args, attr) and getattr(args, attr)


def _parse_cfg(args, driver_name):
    """parse configuration file. If the option is directly given to the command line,
       the option is processed prior to the configuration file."""
    if _is_valid_attr(args, 'config'):
        config = configparser.ConfigParser()
        config.read(args.config)
        # if section is given, verify given section
        if _is_valid_attr(args, 'section'):
            if not config.has_section(args.section):
                raise AssertionError('configuration file must have \'' + driver_name +
                                     '\' section')
            for key in config[args.section]:
                if not _is_valid_attr(args, key):
                    setattr(args, key, config[args.section][key])
        # if section is not given, section name is same with its driver name
        else:
            if not config.has_section(driver_name):
                raise AssertionError('configuration file must have \'' + driver_name +
                                     '\' section')
            secton_to_run = driver_name
            for key in config[secton_to_run]:
                if not _is_valid_attr(args, key):
                    setattr(args, key, config[secton_to_run][key])


def _make_tf2tfliteV2_cmd(args, driver_path, input_path, output_path):
    """make a command for running tf2tfliteV2.py"""
    cmd = [sys.executable, os.path.expanduser(driver_path)]
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
    # optimization pass
    if _is_valid_attr(args, 'all'):
        cmd.append('--all')
    if _is_valid_attr(args, 'convert_nchw_to_nhwc'):
        cmd.append('--convert_nchw_to_nhwc')
    if _is_valid_attr(args, 'fold_dequantize'):
        cmd.append('--fold_dequantize')
    if _is_valid_attr(args, 'fuse_add_with_tconv'):
        cmd.append('--fuse_add_with_tconv')
    if _is_valid_attr(args, 'fuse_batchnorm_with_tconv'):
        cmd.append('--fuse_batchnorm_with_tconv')
    if _is_valid_attr(args, 'fuse_bcq'):
        cmd.append('--fuse_bcq')
    if _is_valid_attr(args, 'fuse_preactivation_batchnorm'):
        cmd.append('--fuse_preactivation_batchnorm')
    if _is_valid_attr(args, 'make_batchnorm_gamma_positive'):
        cmd.append('--make_batchnorm_gamma_positive')
    if _is_valid_attr(args, 'fuse_activation_function'):
        cmd.append('--fuse_activation_function')
    if _is_valid_attr(args, 'fuse_instnorm'):
        cmd.append('--fuse_instnorm')
    if _is_valid_attr(args, 'replace_cw_mul_add_with_depthwise_conv'):
        cmd.append('--replace_cw_mul_add_with_depthwise_conv')
    if _is_valid_attr(args, 'remove_redundant_transpose'):
        cmd.append('--remove_redundant_transpose')
    if _is_valid_attr(args, 'remove_unnecessary_reshape'):
        cmd.append('--remove_unnecessary_reshape')
    if _is_valid_attr(args, 'remove_unnecessary_slice'):
        cmd.append('--remove_unnecessary_slice')
    if _is_valid_attr(args, 'remove_unnecessary_split'):
        cmd.append('--remove_unnecessary_split')
    if _is_valid_attr(args, 'resolve_customop_add'):
        cmd.append('--resolve_customop_add')
    if _is_valid_attr(args, 'resolve_customop_batchmatmul'):
        cmd.append('--resolve_customop_batchmatmul')
    if _is_valid_attr(args, 'resolve_customop_matmul'):
        cmd.append('--resolve_customop_matmul')
    if _is_valid_attr(args, 'shuffle_weight_to_16x1float32'):
        cmd.append('--shuffle_weight_to_16x1float32')
    if _is_valid_attr(args, 'substitute_pack_to_reshape'):
        cmd.append('--substitute_pack_to_reshape')

    return cmd


def _print_version_and_exit(file_path):
    """print version of the file located in the file_path"""
    script_path = os.path.realpath(file_path)
    dir_path = os.path.dirname(script_path)
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    # run one-version
    subprocess.call([os.path.join(dir_path, 'one-version'), script_name])
    sys.exit()
