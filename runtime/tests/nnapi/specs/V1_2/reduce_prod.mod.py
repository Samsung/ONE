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
  model = Model().Operation("REDUCE_PROD", input0, axes, keep_dims).To(output0)
  Example({
      input0: input_data,
      output0: output_data,
  }, model=model).AddVariations("relaxed", "float16")

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{3, 2}"),
    input_data=[-1, -2,
                3, 4,
                5, -6],
    axes=[-1],
    keep_dims=False,
    output0=Output("output0", "TENSOR_FLOAT32", "{3}"),
    output_data=[-1 * -2, 3 * 4, 5 * -6],
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
    input_data=[1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                9.0,  1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60,
                1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40],
    axes=[1, 0, -3, -3],
    keep_dims=False,
    output0=Output("output0", "TENSOR_FLOAT32", "{2}"),
    output_data=[3.16234143225e+4, 1.9619905536e+4],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{4, 3, 2}"),
    input_data=[1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                9.0,  1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60,
                1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40],
    axes=[0, 2],
    keep_dims=True,
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 3, 1}"),
    output_data=[7.74592e+2, 1.197504e+3, 6.6889152e+2],
)
