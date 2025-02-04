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

def test(name, input0, input1, output0, input0_data, input1_data, output_data):
  model = Model().Operation("MINIMUM", input0, input1).To(output0)

  quant8 = DataTypeConverter().Identify({
      input0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
      input1: ["TENSOR_QUANT8_ASYMM", 1.0, 100],
      output0: ["TENSOR_QUANT8_ASYMM", 2.0, 80],
  })

  Example({
      input0: input0_data,
      input1: input1_data,
      output0: output_data,
  }, model=model, name=name).AddVariations("relaxed", "float16", "int32", quant8)


test(
    name="simple",
    input0=Input("input0", "TENSOR_FLOAT32", "{3, 1, 2}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{3, 1, 2}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{3, 1, 2}"),
    input0_data=[1.0, 0.0, -1.0, 11.0, -2.0, -1.44],
    input1_data=[-1.0, 0.0, 1.0, 12.0, -3.0, -1.43],
    output_data=[-1.0, 0.0, -1.0, 11.0, -3.0, -1.44],
)

test(
    name="broadcast",
    input0=Input("input0", "TENSOR_FLOAT32", "{3, 1, 2}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{2}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{3, 1, 2}"),
    input0_data=[1.0, 0.0, -1.0, -2.0, -1.44, 11.0],
    input1_data=[0.5, 2.0],
    output_data=[0.5, 0.0, -1.0, -2.0, -1.44, 2.0],
)


# Test overflow and underflow.
input0 = Input("input0", "TENSOR_QUANT8_ASYMM", "{2}, 1.0f, 128")
input1 = Input("input1", "TENSOR_QUANT8_ASYMM", "{2}, 1.0f, 128")
output0 = Output("output0", "TENSOR_QUANT8_ASYMM", "{2}, 0.5f, 128")
model = Model().Operation("MINIMUM", input0, input1).To(output0)

Example({
    input0: [60, 128],
    input1: [128, 200],
    output0: [0, 128],
}, model=model, name="overflow")
