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

# This script tests the values of weights/const of uint8 quantized model
# Run dalgona with
# --input_model : float circle model
# --input_data : any data
# --analysis : this script
# --analysis_args : path to the h5 dumped data (fake_quantized model)

import os
import sys

if not hasattr(sys, 'argv'):
    sys.argv = ['']
import numpy as np
import math
import json

import QuantizationUtils as q


class GenGoldenWeights(object):
    def recordInstanceNormConst(self, tensor):
        tensor_name = tensor['name'].replace('/', '_')[-255:]
        self._instnorm_const[tensor_name] = tensor['data']

    def recordPReluAlpha(self, tensor):
        tensor_name = tensor['name'].replace('/', '_')[-255:]
        self._alphas[tensor_name] = tensor['data']

    def recordWeights(self, tensor):
        tensor_name = tensor['name'].replace('/', '_')[-255:]
        self._weights[tensor_name] = tensor['data']

    def recordConst(self, tensor):
        tensor_name = tensor['name'].replace('/', '_')[-255:]
        self._const[tensor_name] = tensor['data']

    def genGoldenWeights(self, fq_path, q_path, tensor_name):
        weights = self._weights[tensor_name]

        if self._granularity == 'layer':
            w_min = np.min(weights)
            w_max = np.max(weights)
            w_dfp, w_scale, zp, w_min_ng, w_max_ng = q.float2qint_asym(
                weights, w_min, w_max, bit=8, unsigned=True)
            w_fq = q.qint2float_asym(w_dfp, w_min, w_max, bit=8, unsigned=True)

            # Save quantized weights
            w_quantized = q.float2qint_weights_quant(
                w_fq, w_min_ng, w_scale, dtype=np.uint8)
            json_out = json.dumps(
                {
                    "weights": w_quantized.tolist(),
                    "scale": w_scale.tolist(),
                    "zero_point": zp.tolist(),
                    "min": w_min_ng.tolist(),
                    "max": w_max_ng.tolist()
                },
                indent=2)
            with open(q_path + "/" + tensor_name + ".json", 'w') as f:
                f.write(json_out)

            # Save fake-quantized weights
            json_out = json.dumps({"weights": w_fq.tolist()}, indent=2)
            with open(fq_path + "/" + tensor_name + ".json", 'w') as f:
                f.write(json_out)

        elif self._granularity == 'channel':
            w_min = []
            w_max = []
            if self._dw:
                for i in range(0, weights.shape[3]):
                    w_min.append(np.min(weights[:, :, :, i]))
                    w_max.append(np.max(weights[:, :, :, i]))
            else:
                for i in range(0, weights.shape[0]):
                    w_min.append(np.min(weights[i]))
                    w_max.append(np.max(weights[i]))
            w_min = np.asarray(w_min)
            w_max = np.asarray(w_max)

            w_dfp, w_scale, zp, w_min_ng, w_max_ng = q.float2qint_cw_asym(
                weights, w_min, w_max, bit=8, unsigned=True, dw=self._dw)
            w_fq = q.qint2float_cw_asym(
                w_dfp, w_min, w_max, bit=8, unsigned=True, dw=self._dw)

            # Save quantized weights
            json_out = json.dumps(
                {
                    "weights": w_dfp.tolist(),
                    "scale": w_scale.tolist(),
                    "zero_point": zp.tolist(),
                    "min": w_min_ng.tolist(),
                    "max": w_max_ng.tolist()
                },
                indent=2)
            with open(q_path + "/" + tensor_name + ".json", 'w') as f:
                f.write(json_out)

            # Save fake-quantized weights
            json_out = json.dumps({"weights": w_fq.tolist()}, indent=2)
            with open(fq_path + "/" + tensor_name + ".json", 'w') as f:
                f.write(json_out)

    def genGoldenConst(self, q_path, tensor_name):
        const = self._const[tensor_name]

        w_min = np.min(const)
        w_max = np.max(const)
        w_quantized, w_scale, zp, w_min_ng, w_max_ng = q.float2qint_asym(
            const, w_min, w_max, bit=8, unsigned=True)

        # Save quantized const
        json_out = json.dumps(
            {
                "weights": w_quantized.tolist(),
                "scale": w_scale.tolist(),
                "zero_point": zp.tolist()
            },
            indent=2)
        with open(q_path + "/" + tensor_name + ".json", 'w') as f:
            f.write(json_out)

    def genGoldenPReluAlpha(self, q_path, tensor_name):
        const = self._alphas[tensor_name]
        const_shape = const.shape
        const = const.flatten()

        data = []
        scale = []
        zp = []

        for c in const:
            if c >= 0:
                data.append(1)
                scale.append(c)
                zp.append(0)
            else:
                data.append(0)
                scale.append(-c)
                zp.append(1)

        data = np.asarray(data, dtype=np.uint8).reshape(const_shape)
        scale = np.asarray(scale, dtype=np.float32)
        zp = np.asarray(zp, dtype=np.int64)

        # Save quantized const
        json_out = json.dumps(
            {
                "weights": data.tolist(),
                "scale": scale.tolist(),
                "zero_point": zp.tolist()
            },
            indent=2)

        with open(q_path + "/" + tensor_name + ".json", 'w') as f:
            f.write(json_out)

    def genGoldenInstanceNormConst(self, q_path, tensor_name):
        const = self._instnorm_const[tensor_name]
        const_shape = const.shape
        const = const.flatten()

        data = []
        scale = []
        zp = []

        for c in const:
            if c >= 0:
                data.append(1)
                scale.append(c)
                zp.append(0)
            else:
                data.append(0)
                scale.append(-c)
                zp.append(1)

        data = np.asarray(data, dtype=np.uint8).reshape(const_shape)
        scale = np.asarray(scale, dtype=np.float32)
        zp = np.asarray(zp, dtype=np.int64)

        # Save quantized const
        json_out = json.dumps(
            {
                "weights": data.tolist(),
                "scale": scale.tolist(),
                "zero_point": zp.tolist()
            },
            indent=2)

        with open(q_path + "/" + tensor_name + ".json", 'w') as f:
            f.write(json_out)

    def StartAnalysis(self, args):
        """Called when the analysis starts"""
        print("Analysis started.")
        self._output_path = args
        self._weights = {}
        self._const = {}
        self._alphas = {}
        self._instnorm_const = {}
        self._dw = False
        self._granularity = 'channel'

    def EndAnalysis(self):
        """Called when the analysis ends"""
        model_path = self._output_path
        output_path = model_path + "/expected_outputs"
        fake_quant_output_path = model_path + "/expected_outputs/fake_quantization"
        quant_output_path = model_path + "/expected_outputs/quantization"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        if not os.path.isdir(fake_quant_output_path):
            os.makedirs(fake_quant_output_path)
        if not os.path.isdir(quant_output_path):
            os.makedirs(quant_output_path)

        for tensor_name in self._weights:
            self.genGoldenWeights(fake_quant_output_path, quant_output_path, tensor_name)

        for tensor_name in self._const:
            self.genGoldenConst(quant_output_path, tensor_name)

        for tensor_name in self._alphas:
            self.genGoldenPReluAlpha(quant_output_path, tensor_name)

        for tensor_name in self._instnorm_const:
            self.genGoldenInstanceNormConst(quant_output_path, tensor_name)

    def Conv2DPost(self, name, input, filter, bias, padding, stride, dilation, output,
                   fused_act):
        """Hook for Conv2D node"""
        self.recordWeights(filter)

    def DepthwiseConv2DPost(self, name, input, filter, bias, padding, stride,
                            depth_multiplier, dilation, output, fused_act):
        """Hook for DepthwiseConv2D node"""
        self.recordWeights(filter)
        self._dw = True

    def FullyConnectedPost(self, name, input, weights, bias, output, fused_act):
        """Hook for FullyConnected node"""
        self.recordWeights(weights)

    def TransposeConvPost(self, name, input, filter, output_shape, bias, padding, stride,
                          output):
        """Hook for TransposeConv node"""
        self.recordWeights(filter)

    def InstanceNormPost(self, name, input, gamma, beta, epsilon, output):
        """Hook for InstanceNorm node"""
        assert (gamma['is_const'])
        assert (beta['is_const'])
        if self._granularity == 'channel':
            self.recordInstanceNormConst(gamma)
            self.recordInstanceNormConst(beta)
        else:
            self.recordConst(gamma)
            self.recordConst(beta)

    def DefaultOpPost(self, name, opcode, inputs, output):
        """Default hook called after an operator is executed"""
        if opcode == 'PRelu' and self._granularity == 'channel':
            assert (inputs[1]['is_const'])
            self.recordPReluAlpha(inputs[1])
            return

        for input in inputs:
            if input['is_const']:
                self.recordConst(input)
