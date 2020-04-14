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

def test(input0, output0, axes, keep_dims, input_data, output_data):
  model = Model().Operation("REDUCE_MAX", input0, axes, keep_dims).To(output0)
  quant8 = DataTypeConverter().Identify({
      input0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
      output0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
  })
  Example({
      input0: input_data,
      output0: output_data,
  }, model=model).AddVariations("relaxed", "float16", quant8)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{3, 2}"),
    input_data=[-1, -2,
                3, 4,
                5, -6],
    axes=[-1],
    keep_dims=False,
    output0=Output("output0", "TENSOR_FLOAT32", "{3}"),
    output_data=[-1, 4, 5],
)

# Tests below were adapted from tensorflow/lite/kernels/reduce_test.cc

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1}"),
    input_data=[9.527],
    axes=[0],
    keep_dims=True,
    output0=Output("output0", "TENSOR_FLOAT32", "{1}"),
    output_data=[9.527],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{4, 3, 2}"),
    input_data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
    axes=[1, 0, -3, -3],
    keep_dims=False,
    output0=Output("output0", "TENSOR_FLOAT32", "{2}"),
    output_data=[2.3, 2.4],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{4, 3, 2}"),
    input_data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
    axes=[0, 2],
    keep_dims=True,
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 3, 1}"),
    output_data=[2.0, 2.2, 2.4],
)
