#!/usr/bin/env bash
''''export SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"                  # '''
''''export PY_PATH=${SCRIPT_PATH}/../bin/venv/bin/python                                # '''
''''test -f ${PY_PATH} && exec ${PY_PATH} "$0" "$@"                                     # '''
''''echo "Error: Virtual environment not found. Please run 'one-prepare-venv' command." # '''
''''exit 255                                                                            # '''

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

# NOTE This is an experimental script to evaluate onnx-circle conversion
#      by running onnxruntime and luci-interpreter.
#      Plan is to run this regularly in CI

import subprocess
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--onnx', type=str, required=True)
parser.add_argument('--circle', type=str, required=True)
args = parser.parse_args()

driver = args.driver
onnx_filepath = args.onnx
circle_filepath = args.circle


def to_numpy(tensor):
    return tensor.cpu().numpy()


def to_nhwc(tensor):
    if (tensor.ndim == 4):
        return np.transpose(tensor, (0, 2, 3, 1))
    return tensor


class OnnxRunner:
    def __init__(self, filepath):
        self.filepath = filepath
        self.session = None
        self.inputs = None
        self.inputs_size = None
        self.inputs_data = None
        self.outputs = None
        self.outputs_size = None

    def load(self):
        model = onnx.load(self.filepath)
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        # NOTE this is needed for U18.04
        # referenced: https://github.com/microsoft/onnxruntime/issues/10113
        options.intra_op_num_threads = 4
        # NOTE set `providers` for https://github.com/microsoft/onnxruntime/issues/17631
        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(
            self.filepath, sess_options=options, providers=providers)

    def feed_random_inputs(self):
        self.inputs = self.session.get_inputs()
        self.inputs_size = len(self.inputs)
        # reset input dictionary
        self.inputs_data = {}
        for in_idx in range(self.inputs_size):
            input_shape = self.inputs[in_idx].shape
            input_type = self.inputs[in_idx].type
            if input_type == 'tensor(float)':
                torch_type = torch.float32
            else:
                # TODO support other dtype
                raise SystemExit("Unsupported input dtype")

            x = torch.randn(input_shape, dtype=torch_type)
            input_npa = to_numpy(x)
            self.inputs_data.update({self.inputs[in_idx].name: input_npa})

            # save NHWC form of input for luci-interpreter
            input_npa_nhwc = to_nhwc(input_npa)
            input_npa_nhwc.tofile(circle_filepath + ".input" + str(in_idx))

    def run(self):
        self.outs = self.session.run(None, self.inputs_data)

    def get_outputs(self):
        self.outputs = self.session.get_outputs()
        self.outputs_size = len(self.outputs)


# Run ONNX model
print("Run ONNX...")
onnx_runner = OnnxRunner(onnx_filepath)
onnx_runner.load()
onnx_runner.feed_random_inputs()
onnx_runner.run()
onnx_runner.get_outputs()

# Execute luci interpreter
print("Run luci-interpreter...")
process = subprocess.run(
    [
        driver, circle_filepath,
        str(onnx_runner.inputs_size), circle_filepath + ".input",
        circle_filepath + ".output"
    ],
    check=True)

# Compare results
rtolerance = 1e-03
atolerance = 1e-04
result_compare = True
for idx in range(onnx_runner.outputs_size):
    output_shape = onnx_runner.outputs[idx].shape
    output_type = onnx_runner.outputs[idx].type
    if output_type == 'tensor(float)':
        output_np_type = np.float32
    else:
        # TODO support other dtype
        raise SystemExit("Unsupported output dtype")

    # output of luci-interpreter
    output_data = np.fromfile(circle_filepath + ".output" + str(idx), output_np_type)
    shape_file = open(circle_filepath + ".output" + str(idx) + ".shape", 'r')
    output_shape = [int(i) for i in shape_file.read().split(',')]
    luci_output_data = np.reshape(output_data, output_shape)

    # output of onnx runtime
    output_nchw = onnx_runner.outs[idx]
    output_nhwc = to_nhwc(output_nchw)

    # diff has tensor of boolean for each values within tolerance or not
    diff = np.isclose(output_nhwc, luci_output_data, rtol=rtolerance, atol=atolerance)
    # get one boolean if all are True then True
    result_compare_one = np.all(diff)
    print("Compare", idx, result_compare_one)
    if (not result_compare_one):
        diff_val = np.subtract(output_nhwc, luci_output_data)
        print("ONNX Result", output_nhwc)
        print("Diff", diff_val)
        print("Diff Max", np.ndarray.max(diff_val))

    result_compare = result_compare and result_compare_one

if (not result_compare):
    exit(-1)

exit(0)
