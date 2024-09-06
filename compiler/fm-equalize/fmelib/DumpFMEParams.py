#!/usr/bin/env python3
# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

# Script that dumps scale/shift parameters for FM equalization
# NOTE This script runs on dalgona

from typing import Dict, List

import numpy as np
import json
import os

from ComputeParam import getActivationMax


# Recursively visit items and round floats with ndigits
def _pretty_float(item, ndigits=6):
    if isinstance(item, list):
        return [_pretty_float(x, ndigits) for x in item]
    if isinstance(item, float):
        return round(item, ndigits)
    return item


# Dump scale/shift parameters for FM equalization
# The parameters are saved in the json file given as an input
#
# Contents of the json file ('<model_name>.fme_patterns.json')
#
# Before
# [
#   {
#     “front”: <tensor_name>,
#     “back”: <tensor_name>,
#     “type”: ScaleOnly/ShiftOnly/ScaleShift,
#   },
#   …
# ]
#
# After
# [
#   {
#     “front”: <tensor_name>,
#     “back”: <tensor_name>,
#     “type”: ScaleOnly/ShiftOnly/ScaleShift,
#     “scale”: [..],
#     “shift”: [..]
#   },
#   …
# ]
class DumpFMEParams:

    # Return path to the data
    # self._dir_path/<tid>.<data_idx>.npy
    def _data_path(self, tid: int, data_idx: int):
        assert (self._dir_path != None)  # FIX_CALLER_UNLESS
        return self._dir_path + '/' + str(tid) + '.' + str(data_idx) + '.npy'

    def record_activaion_max(self, tensor: dict):
        tensor_name = tensor['name']

        act_max = getActivationMax(tensor['data'])
        if tensor_name not in self.activation_max:
            self.activation_max[tensor_name] = act_max
        else:
            self.activation_max[tensor_name] = np.maximum.reduce(
                [self.activation_max[tensor_name], act_max])

    def StartAnalysis(self, args: str):
        """Called when the analysis starts"""
        self._json_path = args
        self._dir_path = os.path.dirname(args)
        # Data structure to save tensor information
        # {
        #   <tensor_name>: <activation max>
        # }
        self.activation_max: Dict[str, float] = {}

        # Names of target tensors ('back' of equalization pattern)
        self._target_tensors = []

        with open(args, 'r') as f:
            patterns = json.load(f)
            for pattern in patterns:
                self._target_tensors.append(pattern['front'])
            self._patterns = patterns

        # Set of operators with ReLU fused activation
        # This is used to check if negative scales are fused across ReLU
        self._relu_fused_act = set()

    def DefaultOpPost(self, name, opcode, inputs, output):
        # DO NOTHING
        pass

    def Conv2DPost(self, name, input, filter, bias, padding, stride, dilation, output,
                   fused_act):
        if name in self._target_tensors:
            self.record_activaion_max(output)
        # TODO consider activation functions
        # if fused_act == "relu":
        #     self._relu_fused_act.add(name)

    def DepthwiseConv2DPost(self, name, input, filter, bias, padding, stride,
                            depthMultiplier, dilation, output, fused_act):
        if name in self._target_tensors:
            self.record_activaion_max(output)

    def FullyConnectedPost(self, name, input, weights, bias, output, fused_act):
        if name in self._target_tensors:
            self.record_activaion_max(output)

    def InstanceNormPost(self, name, input, gamma, beta, epsilon, output, fused_act):
        if name in self._target_tensors:
            self.record_activaion_max(output)

    def TransposeConvPost(self, name, input, filter, output_shape, bias, padding, stride,
                          output):
        if name in self._target_tensors:
            self.record_activaion_max(output)

    def EndAnalysis(self):
        """Called when the analysis ends"""

        res = []
        for pattern in self._patterns:
            # pattern['front'] is the input of pattern['back']
            tensor_name = pattern['front']
            eq_type = pattern['type']

            act_max = self.activation_max[tensor_name]

            # tensor_data = []
            # for i in range(num_data):
            #     with open(self._data_path(tid, i), 'rb') as f:
            #         tensor_data.append(np.load(f))

            scales = None
            if eq_type == 'ScaleOnly':
                scales = act_max
                pattern['act_scale'] = _pretty_float(scales).tolist()
            else:
                raise ValueError('Unknown equalization type: ' + eq_type)

            # if scales != None:
            #     if tensor_name in self._relu_fused_act:
            #         if any(s < 0 for s in scales):
            #             continue

            res.append(pattern)

        # Overwrite
        with open(self._json_path, 'w') as json_file:
            json.dump(res, json_file)
