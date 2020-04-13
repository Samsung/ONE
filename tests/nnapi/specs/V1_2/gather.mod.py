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

def test(input0, axis, indices, output0, input_data, output_data):
  model = Model().Operation("GATHER", input0, axis, indices).To(output0)

  quant8 = DataTypeConverter().Identify({
      input0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
      output0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
  })

  int32 = DataTypeConverter().Identify({
      input0: ["TENSOR_INT32"],
      output0: ["TENSOR_INT32"],
  })

  float16 = DataTypeConverter().Identify({
      input0: ["TENSOR_FLOAT16"],
      output0: ["TENSOR_FLOAT16"],
  })

  Example({
      input0: input_data,
      output0: output_data,
  }, model=model).AddVariations("relaxed", quant8, int32, float16)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{2, 2}"),
    axis=0,
    indices=[1, 0],
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 2}"),
    input_data=[-2.0, 0.2,
                 0.7, 0.8],
    output_data=[0.7, 0.8,
                -2.0, 0.2],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{2, 2}"),
    axis=0,
    indices=[1], # Unlike TensorFlow, 0-D arguments and outputs are not supported.
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 2}"),
    input_data=[-2.0, 0.2,
                 0.7, 0.8],
    output_data=[0.7, 0.8],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{3}"),
    axis=0,
    indices=[1],
    output0=Output("output0", "TENSOR_FLOAT32", "{1}"),
    input_data=[1, 2, 3],
    output_data=[2],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{3}"),
    axis=0,
    indices=[1, 0],
    output0=Output("output0", "TENSOR_FLOAT32", "{2}"),
    input_data=[1, 2, 3],
    output_data=[2, 1],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 2, 2}"),
    axis=0,
    indices=[0, 0],
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 2, 2}"),
    input_data=[-2.0, 0.2,
                 0.7, 0.8],
    output_data=[-2.0, 0.2,
                  0.7, 0.8,
                 -2.0, 0.2,
                  0.7, 0.8],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{4, 1}"),
    axis=0,
    indices=[1, 3],
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 1}"),
    input_data=[-2.0, 0.2, 0.7, 0.8],
    output_data=[0.2, 0.8],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 2, 3}"),
    axis=1,
    indices=[1, 0],
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 2, 3}"),
    input_data=[1, 2, 3,
                4, 5, 6],
    output_data=[4, 5, 6,
                 1, 2, 3],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 2, 3}"),
    axis=-1,
    indices=[2, 0],
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 2, 2}"),
    input_data=[1, 2, 3,
                4, 5, 6],
    output_data=[3, 1,
                 6, 4],
)
