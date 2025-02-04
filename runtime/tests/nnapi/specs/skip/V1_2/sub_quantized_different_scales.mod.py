#
# Copyright (C) 2018 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import itertools

def dequantize(x, scale, offset):
  return (x - offset) * scale

def quantize(x, scale, offset):
  return max(0, min(255, int(round(x / scale)) + offset))

def create_test(input0_scale, input0_offset,
                input1_scale, input1_offset,
                output_scale, output_offset):
  def sub_quantized(a, b):
    a_dequantized = dequantize(a, input0_scale, input0_offset)
    b_dequantized = dequantize(b, input1_scale, input1_offset)
    return quantize(a_dequantized - b_dequantized, output_scale, output_offset)

  values = [0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255]
  inputs = list(itertools.product(values, values))
  input0_values, input1_values = zip(*inputs)
  output_values = [sub_quantized(a, b) for a, b in inputs]
  size = len(output_values)
  input0 = Input("input0", "TENSOR_QUANT8_ASYMM",
                 "{%d}, %g, %d" % (size, input0_scale, input0_offset))
  input1 = Input("input1", "TENSOR_QUANT8_ASYMM",
                 "{%d}, %g, %d" % (size, input1_scale, input1_offset))
  activation = 0
  output0 = Output("output0", "TENSOR_QUANT8_ASYMM",
                   "{%d}, %g, %d" % (size, output_scale, output_offset))
  model = Model().Operation("SUB", input0, input1, activation).To(output0)
  Example({
      input0: input0_values,
      input1: input1_values,
      output0: output_values,
  })

scales_and_offsets = [(1.0, 0),
                      (1.0, 1),
                      (0.01, 120),
                      (10.0, 120)]
for params in itertools.product(scales_and_offsets,
                                scales_and_offsets,
                                scales_and_offsets):
  input0_params, input1_params, output_params = params
  create_test(*input0_params, *input1_params, *output_params)
