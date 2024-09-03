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

import argparse
from circle_schema import circle
import h5py
import numpy as np
import os
import sys

h5dtypes = {
    "float32": ">f4",
    "uint8": "u1",
    "int8": "i1",
    "bool": "u1",
    "int32": "int32",
    "int64": "int64"
}


def quantize(h5_path, circle_path, h5_out_path):
    with open(circle_path, 'rb') as f:
        graph = circle.Model.GetRootAsModel(f.read(), 0).Subgraphs(0)
    input_tensor = graph.Tensors(graph.Inputs(0))
    input_names = [input_tensor.Name()]

    with h5py.File(h5_path, 'r') as hf:
        dset = hf['/value/0']
        arr = np.array(dset)

    if not np.issubdtype(arr.dtype,
                         np.float32) or input_tensor.Type() != circle.TensorType.UINT8:
        print("Not f32 to q8u")
        sys.exit(-1)

    # copied from python-tools/examples/pytorch_tutorial/main.py
    dtype = 'uint8'

    def _quantize_input0(data):
        qparam = graph.Tensors(graph.Inputs(0)).Quantization()
        rescaled_data = data / qparam.ScaleAsNumpy()[0] + qparam.ZeroPointAsNumpy()[0]
        return np.round(rescaled_data).clip(np.iinfo(dtype).min,
                                            np.iinfo(dtype).max).astype(dtype)

    qarr = _quantize_input0(arr)

    ensure_output_dir(h5_out_path)
    with h5py.File(h5_out_path, 'w') as hf:
        name_grp = hf.create_group("name")
        val_grp = hf.create_group("value")
        idx = 0
        val_grp.create_dataset(str(idx), data=qarr, dtype=h5dtypes[dtype])
        name_grp.attrs[str(idx)] = input_names[0]


def dequantize(h5_path, circle_path, h5_out_path):
    with open(circle_path, 'rb') as f:
        graph = circle.Model.GetRootAsModel(f.read(), 0).Subgraphs(0)
    output_tensor = graph.Tensors(graph.Outputs(0))
    output_names = [output_tensor.Name()]

    with h5py.File(h5_path, 'r') as hf:
        dset = hf['/value/0']
        arr = np.array(dset)
    if not np.issubdtype(arr.dtype,
                         np.uint8) or output_tensor.Type() != circle.TensorType.UINT8:
        print("Not q8u to f32")
        sys.exit(-1)

    # copied from python-tools/examples/pytorch_tutorial/main.py
    def _dequantize_output0(data):
        qparam = graph.Tensors(graph.Outputs(0)).Quantization()
        return (data.astype(np.float32) -
                qparam.ZeroPointAsNumpy()[0]) * qparam.ScaleAsNumpy()[0]

    qarr = _dequantize_output0(arr)

    ensure_output_dir(h5_out_path)
    with h5py.File(h5_out_path, 'w') as hf:
        name_grp = hf.create_group("name")
        val_grp = hf.create_group("value")
        idx = 0
        val_grp.create_dataset(str(idx), data=qarr, dtype='>f4')
        name_grp.attrs[str(idx)] = output_names[0]


def makeArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5',
                        type=str,
                        help='path to h5 file either input or output to model')
    parser.add_argument('circle', type=str, help='path to quantized circle model')
    parser.add_argument('-o',
                        '--output',
                        action='store',
                        dest="out_path",
                        help="output file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-q',
        '--quantize',
        action='store_true',
        help="quantize f32 to q8u using circle input's qparam (default: false)")
    group.add_argument(
        '-d',
        '--dequantize',
        action='store_true',
        help="dequantize q8u to f32 using circle output's qparam (default: false)")
    return parser


def parseArgs():
    args = parser.parse_args()
    return args


def ensure_output_dir(out_path):
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)


if __name__ == '__main__':
    parser = makeArgParser()
    args = parseArgs()

    h5_path, circle_path = args.h5, args.circle

    if args.out_path:
        out_path = args.out_path
    else:
        h5_name, ext = os.path.splitext(h5_path)
        out_path = h5_name + '_' + ext

    if args.quantize:
        quantize(h5_path, circle_path, out_path)

    if args.dequantize:
        dequantize(h5_path, circle_path, out_path)
