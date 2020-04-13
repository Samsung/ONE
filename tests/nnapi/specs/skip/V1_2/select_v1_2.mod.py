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
def test(name, input0, input1, input2, output0, input0_data, input1_data, input2_data, output_data):
  model = Model().Operation("SELECT", input0, input1, input2).To(output0)
  quant8 = DataTypeConverter().Identify({
      input1: ["TENSOR_QUANT8_ASYMM", 1.5, 129],
      input2: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
      output0: ["TENSOR_QUANT8_ASYMM", 1.0, 128],
  })
  example = Example({
      input0: input0_data,
      input1: input1_data,
      input2: input2_data,
      output0: output_data,
  }, model=model, name=name).AddVariations("int32", "float16", "relaxed", quant8)

test(
    name="one_dim",
    input0=Input("input0", "TENSOR_BOOL8", "{3}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{3}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{3}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{3}"),
    input0_data=[True, False, True],
    input1_data=[1, 2, 3],
    input2_data=[4, 5, 6],
    output_data=[1, 5, 3],
)

test(
    name="two_dim",
    input0=Input("input0", "TENSOR_BOOL8", "{2, 2}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{2, 2}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{2, 2}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 2}"),
    input0_data=[False, True, False, True],
    input1_data=[1, 2, 3, 4],
    input2_data=[5, 6, 7, 8],
    output_data=[5, 2, 7, 4],
)

test(
    name="five_dim",
    input0=Input("input0", "TENSOR_BOOL8", "{2, 1, 2, 1, 2}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{2, 1, 2, 1, 2}"),
    input2=Input("input2", "TENSOR_FLOAT32", "{2, 1, 2, 1, 2}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 1, 2, 1, 2}"),
    input0_data=[True, False, True, False, True, False, True, False],
    input1_data=[1, 2, 3, 4, 5, 6, 7, 8],
    input2_data=[9, 10, 11, 12, 13, 14, 15, 16],
    output_data=[1, 10, 3, 12, 5, 14, 7, 16],
)
