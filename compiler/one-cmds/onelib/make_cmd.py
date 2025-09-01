#!/usr/bin/env python

# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os
import sys

import onelib.constant as _constant


def is_valid_attr(args, attr):
    return hasattr(args, attr) and getattr(args, attr)


def make_tf2tfliteV2_cmd(args, driver_path, input_path, output_path):
    """make a command for running tf2tfliteV2.py"""
    cmd = [sys.executable, os.path.expanduser(driver_path)]
    # verbose
    if is_valid_attr(args, 'verbose'):
        cmd.append('--verbose')
    # model_format
    if is_valid_attr(args, 'model_format_cmd'):
        cmd.append(getattr(args, 'model_format_cmd'))
    elif is_valid_attr(args, 'model_format'):
        cmd.append('--' + getattr(args, 'model_format'))
    else:
        cmd.append('--graph_def')  # default value
    # input_path
    if is_valid_attr(args, 'input_path'):
        cmd.append('--input_path')
        cmd.append(os.path.expanduser(input_path))
    # output_path
    if is_valid_attr(args, 'output_path'):
        cmd.append('--output_path')
        cmd.append(os.path.expanduser(output_path))
    # input_arrays
    if is_valid_attr(args, 'input_arrays'):
        cmd.append('--input_arrays')
        cmd.append(getattr(args, 'input_arrays'))
    # input_shapes
    if is_valid_attr(args, 'input_shapes'):
        cmd.append('--input_shapes')
        cmd.append(getattr(args, 'input_shapes'))
    # output_arrays
    if is_valid_attr(args, 'output_arrays'):
        cmd.append('--output_arrays')
        cmd.append(getattr(args, 'output_arrays'))

    # experimental options
    if is_valid_attr(args, 'experimental_disable_batchmatmul_unfold'):
        cmd.append('--experimental_disable_batchmatmul_unfold')

    return cmd


def make_tflite2circle_cmd(driver_path, input_path, output_path):
    """make a command for running tflite2circle"""
    cmd = [driver_path, input_path, output_path]
    return [os.path.expanduser(c) for c in cmd]


def make_circle2circle_cmd(args, driver_path, input_path, output_path):
    """make a command for running circle2circle"""
    cmd = [os.path.expanduser(c) for c in [driver_path, input_path, output_path]]
    # profiling
    if is_valid_attr(args, 'generate_profile_data'):
        cmd.append('--generate_profile_data')
    # optimization pass(only true/false options)
    # TODO support options whose number of arguments is more than zero
    for opt in _constant.CONSTANT.OPTIMIZATION_OPTS:
        if is_valid_attr(args, opt[0]):
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
