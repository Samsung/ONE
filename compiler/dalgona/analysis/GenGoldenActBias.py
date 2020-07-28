# Copyright 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script tests the values/scale/zero_point of uint8 quantized activation/bias
# Run dalgona with
# --input_model : fake_quantized circle model
# --input_data : representatitve dataset
# --analysis : this script
# --analysis_args : path to the h5 dumped data

import os
import sys
import json

if not hasattr(sys, 'argv'):
    sys.argv = ['']
import numpy as np
import math

import QuantizationUtils as q


class GenGoldenActBias(object):
    def recordInputs(self, input_list):
        self._inputs.append(input_list)

    def recordBiasInputWeights(self, bias, input, weights):
        bias_name = bias['name'].replace('/', '_')[-255:]
        input_name = input['name'].replace('/', '_')[-255:]
        weights_name = weights['name'].replace('/', '_')[-255:]
        self._biw[bias_name] = (input_name, weights_name)

    def recordConst(self, tensor):
        tensor_name = tensor['name'].replace('/', '_')[-255:]
        self._const[tensor_name] = tensor

    def recordMinMax(self, tensor):
        tensor_name = tensor['name'].replace('/', '_')[-255:]
        if tensor_name in self._minmax_dict:
            self._minmax_dict[tensor_name]['min'].append(np.min(tensor['data']))
            self._minmax_dict[tensor_name]['max'].append(np.max(tensor['data']))
        else:
            self._minmax_dict[tensor_name] = {
                'min': [np.min(tensor['data'])],
                'max': [np.max(tensor['data'])]
            }

    def getMinMax(self, tensor_name, mode):
        if mode == 'percentile':
            min = np.percentile(self._minmax_dict[tensor_name]['min'], 1)
            max = np.percentile(self._minmax_dict[tensor_name]['max'], 99)
            return (min, max)
        elif mode == 'average':
            min = np.average(self._minmax_dict[tensor_name]['min'])
            max = np.average(self._minmax_dict[tensor_name]['max'])
            return (min, max)
        elif mode == 'moving_avg':
            batch_size = 16
            alpha = 0.9
            curr_avg_min = np.min(self._minmax_dict[tensor_name]['min'][0:batch_size])
            curr_avg_max = np.max(self._minmax_dict[tensor_name]['max'][0:batch_size])
            i = batch_size
            while i < len(self._minmax_dict[tensor_name]['min']):
                batch_min = np.min(
                    self._minmax_dict[tensor_name]['min'][i:i + batch_size])
                batch_max = np.max(
                    self._minmax_dict[tensor_name]['max'][i:i + batch_size])
                curr_avg_min = alpha * curr_avg_min + (1 - alpha) * batch_min
                curr_avg_max = alpha * curr_avg_max + (1 - alpha) * batch_max
                i += batch_size
            return (curr_avg_min, curr_avg_max)
        elif mode == 'exact':
            min = np.min(self._minmax_dict[tensor_name]['min'])
            max = np.max(self._minmax_dict[tensor_name]['max'])
            return (min, max)
        else:
            raise SystemExit("Unsupported mode")

    def genGoldenBias(self, q_path, tensor_name, exp_scale, granularity):
        data = self._const[tensor_name]['data']
        q_data = q.float2qint_bias(data, exp_scale)
        if granularity == 'layer':
            # Save quantized bias
            json_out = json.dumps(
                {
                    "weights": q_data.tolist(),
                    "scale": float(exp_scale)
                }, indent=2)
            with open(q_path + "/" + tensor_name + ".json", 'w') as f:
                f.write(json_out)

        elif granularity == 'channel':
            # Save quantized bias
            json_out = json.dumps(
                {
                    "weights": q_data.tolist(),
                    "scale": exp_scale.tolist()
                }, indent=2)
            with open(q_path + "/" + tensor_name + ".json", 'w') as f:
                f.write(json_out)

    def collectWeightsScales(self, tensor_name, granularity):
        if granularity == 'layer':
            data = self._const[tensor_name]['data']
            exp_scale = self._const[tensor_name]['quantparam']['scale'][0]
            self._scales[tensor_name] = exp_scale

        elif granularity == 'channel':
            data = self._const[tensor_name]['data']
            exp_scale = self._const[tensor_name]['quantparam']['scale']
            self._scales[tensor_name] = exp_scale

    def genGoldenActivation(self, minmax_path, q_path, tensor_name):
        rmin, rmax = self.getMinMax(tensor_name, 'percentile')
        min_ng, max_ng, exp_scale, exp_zero_point = q.nudge_asym(rmin, rmax)
        self._scales[tensor_name] = exp_scale

        # Save min/max
        json_out = json.dumps({'min': float(rmin), 'max': float(rmax)}, indent=2)
        with open(minmax_path + "/" + tensor_name + ".json", 'w') as f:
            f.write(json_out)

        # Save scale/zp
        json_out = json.dumps(
            {
                "scale": float(exp_scale),
                "zero_point": float(exp_zero_point)
            }, indent=2)
        with open(q_path + "/" + tensor_name + ".json", 'w') as f:
            f.write(json_out)

    def genInputData(self, input_path):
        for i, inputs in enumerate(self._inputs):
            with open(input_path + "/" + str(i) + ".txt", 'w') as f:
                text = ""
                for input in inputs:
                    text += np.array2string(
                        input.flatten(), separator=',',
                        max_line_width=9999999)[1:-1] + "\n"
                f.write(text)

    def StartAnalysis(self, args):
        """Called when the analysis starts"""
        print("Analysis started.")
        self._output_path = args
        self._minmax_dict = {}
        self._const = {}
        self._bias = []
        self._scales = {}  # for bias calculation
        self._biw = {}  # for bias calculation
        self._inputs = []
        self._granularity = 'channel'

    def EndAnalysis(self):
        """Called when the analysis ends"""
        granularity = self._granularity
        model_path = self._output_path
        output_path = model_path + "/expected_outputs"
        minmax_output_path = model_path + "/expected_outputs/record_minmax"
        quant_output_path = model_path + "/expected_outputs/quantization"
        input_path = model_path + "/test_inputs"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        if not os.path.isdir(minmax_output_path):
            os.makedirs(minmax_output_path)
        if not os.path.isdir(quant_output_path):
            os.makedirs(quant_output_path)
        if not os.path.isdir(input_path):
            os.makedirs(input_path)

        self.genInputData(input_path)

        for tensor_name in self._minmax_dict:
            self.genGoldenActivation(minmax_output_path, quant_output_path, tensor_name)

        # This is for bias quantization.
        for tensor_name in self._const:
            if tensor_name not in self._biw:
                self.collectWeightsScales(tensor_name, granularity)

        for tensor_name in self._biw:
            input, weights = self._biw[tensor_name]
            S_i = self._scales[input]
            S_w = np.asarray(self._scales[weights])
            self.genGoldenBias(quant_output_path, tensor_name, S_i * S_w, granularity)

    def StartNetworkExecution(self, inputs):
        """Called when the execution of a network starts"""
        input_data = []
        for input in inputs:
            self.recordMinMax(input)
            input_data.append(input['data'])
        self.recordInputs(input_data)

    def DefaultOpPost(self, name, opcode, inputs, output):
        """Default hook called after an operator is executed"""
        self.recordMinMax(output)

    def Conv2DPost(self, name, input, filter, bias, padding, stride, dilation, output,
                   fused_act):
        """Hook for Conv2D node"""
        self.recordConst(filter)
        self.recordConst(bias)
        self.recordBiasInputWeights(bias, input, filter)
        self.recordMinMax(output)

    def DepthwiseConv2DPost(self, name, input, filter, bias, padding, stride,
                            depth_multiplier, dilation, output, fused_act):
        """Hook for DepthwiseConv2D node"""
        self.recordConst(filter)
        self.recordConst(bias)
        self.recordBiasInputWeights(bias, input, filter)
        self.recordMinMax(output)

    def FullyConnectedPost(self, name, input, weights, bias, output, fused_act):
        """Hook for FullyConnected node"""
        self.recordConst(weights)
        self.recordConst(bias)
        self.recordBiasInputWeights(bias, input, weights)
        self.recordMinMax(output)

    def TransposeConvPost(self, name, input, filter, output_shape, bias, padding, stride,
                          output):
        """Hook for TransposeConv node"""
        self.recordConst(filter)
        self.recordConst(bias)
        self.recordBiasInputWeights(bias, input, filter)
        self.recordMinMax(output)

    def SplitPost(self, name, split_dim, input, num_split, outputs):
        """Hook for Split node"""
        for output in outputs:
            self.recordMinMax(output)
