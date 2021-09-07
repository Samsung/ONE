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


class _CONSTANT:
    __slots__ = ()  # This prevents access via __dict__.
    OPTIMIZATION_OPTS = (
        # (OPTION_NAME, HELP_MESSAGE)
        ('O1', 'enable O1 optimization pass'),
        ('convert_nchw_to_nhwc',
         'Experimental: This will convert NCHW operators to NHWC under the assumption that input model is NCHW.'
         ),
        ('nchw_to_nhwc_input_shape',
         'convert the input shape of the model (argument for convert_nchw_to_nhwc)'),
        ('nchw_to_nhwc_output_shape',
         'convert the output shape of the model (argument for convert_nchw_to_nhwc)'),
        ('fold_add_v2', 'fold AddV2 op with constant inputs'),
        ('fold_cast', 'fold Cast op with constant input'),
        ('fold_dequantize', 'fold Dequantize op'),
        ('fold_dwconv', 'fold Depthwise Convolution op with constant inputs'),
        ('fold_sparse_to_dense', 'fold SparseToDense op'),
        ('forward_reshape_to_unaryop', 'Forward Reshape op'),
        ('fuse_add_with_tconv', 'fuse Add op to Transposed'),
        ('fuse_batchnorm_with_conv', 'fuse BatchNorm op to Convolution op'),
        ('fuse_batchnorm_with_dwconv', 'fuse BatchNorm op to Depthwise Convolution op'),
        ('fuse_batchnorm_with_tconv', 'fuse BatchNorm op to Transposed Convolution op'),
        ('fuse_bcq', 'apply Binary Coded Quantization'),
        ('fuse_preactivation_batchnorm',
         'fuse BatchNorm operators of pre-activations to Convolution op'),
        ('fuse_mean_with_mean', 'fuse two consecutive Mean ops'),
        ('fuse_transpose_with_mean',
         'fuse Mean with a preceding Transpose under certain conditions'),
        ('make_batchnorm_gamma_positive',
         'make negative gamma of BatchNorm to a small positive value (1e-10).'
         ' Note that this pass can change the execution result of the model.'
         ' So, use it only when the impact is known to be acceptable.'),
        ('fuse_activation_function', 'fuse Activation function to a preceding operator'),
        ('fuse_instnorm', 'fuse ops to InstanceNorm operator'),
        ('replace_cw_mul_add_with_depthwise_conv',
         'replace channel-wise Mul/Add with DepthwiseConv2D'),
        ('remove_fakequant', 'remove FakeQuant ops'),
        ('remove_quantdequant', 'remove Quantize-Dequantize sequence'),
        ('remove_redundant_reshape', 'fuse or remove subsequent Reshape ops'),
        ('remove_redundant_transpose', 'fuse or remove subsequent Transpose ops'),
        ('remove_unnecessary_reshape', 'remove unnecessary reshape ops'),
        ('remove_unnecessary_slice', 'remove unnecessary slice ops'),
        ('remove_unnecessary_strided_slice', 'remove unnecessary strided slice ops'),
        ('remove_unnecessary_split', 'remove unnecessary split ops'),
        ('resolve_customop_add', 'convert Custom(Add) op to Add op'),
        ('resolve_customop_batchmatmul',
         'convert Custom(BatchMatmul) op to BatchMatmul op'),
        ('resolve_customop_matmul', 'convert Custom(Matmul) op to Matmul op'),
        ('resolve_customop_max_pool_with_argmax',
         'convert Custom(MaxPoolWithArgmax) to net of builtin operators'),
        ('shuffle_weight_to_16x1float32',
         'convert weight format of FullyConnected op to SHUFFLED16x1FLOAT32.'
         ' Note that it only converts weights whose row is a multiple of 16'),
        ('substitute_pack_to_reshape', 'convert single input Pack op to Reshape op'),
        ('substitute_padv2_to_pad', 'convert certain condition PadV2 to Pad'),
        ('substitute_squeeze_to_reshape', 'convert certain condition Squeeze to Reshape'),
        ('substitute_strided_slice_to_reshape',
         'convert certain condition StridedSlice to Reshape'),
        ('substitute_transpose_to_reshape',
         'convert certain condition Transpose to Reshape'),
        ('transform_min_max_to_relu6', 'transform Minimum-Maximum pattern to Relu6 op'),
        ('transform_min_relu_to_relu6', 'transform Minimum(6)-Relu pattern to Relu6 op'))


_CONSTANT = _CONSTANT()


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


def _is_valid_attr(args, attr):
    return hasattr(args, attr) and getattr(args, attr)


def _parse_cfg(args, driver_name):
    """parse configuration file. If the option is directly given to the command line,
       the option is processed prior to the configuration file."""
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
    for opt in _CONSTANT.OPTIMIZATION_OPTS:
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
