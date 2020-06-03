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
def test(name, input0, input1, output0, input0_data, input1_data, output_data, do_variations=True):
  model = Model().Operation("LESS_EQUAL", input0, input1).To(output0)
  example = Example({
      input0: input0_data,
      input1: input1_data,
      output0: output_data,
  }, model=model, name=name)
  if do_variations:
      example.AddVariations("int32", "float16", "relaxed")

test(
    name="simple",
    input0=Input("input0", "TENSOR_FLOAT32", "{3}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{3}"),
    output0=Output("output0", "TENSOR_BOOL8", "{3}"),
    input0_data=[5, 7, 10],
    input1_data=[10, 7, 5],
    output_data=[True, True, False],
)

test(
    name="broadcast",
    input0=Input("input0", "TENSOR_FLOAT32", "{2, 1}"),
    input1=Input("input1", "TENSOR_FLOAT32", "{2}"),
    output0=Output("output0", "TENSOR_BOOL8", "{2, 2}"),
    input0_data=[5, 10],
    input1_data=[10, 5],
    output_data=[True, True, True, False],
)

test(
    name="quantized_different_scale",
    input0=Input("input0", ("TENSOR_QUANT8_ASYMM", [3], 1.0, 128)),
    input1=Input("input1", ("TENSOR_QUANT8_ASYMM", [1], 2.0, 128)),
    output0=Output("output0", "TENSOR_BOOL8", "{3}"),
    input0_data=[129, 130, 131], # effectively 1, 2, 3
    input1_data=[129],           # effectively 2
    output_data=[True, True, False],
    do_variations=False,
)

test(
    name="quantized_different_zero_point",
    input0=Input("input0", ("TENSOR_QUANT8_ASYMM", [3], 1.0, 128)),
    input1=Input("input1", ("TENSOR_QUANT8_ASYMM", [1], 1.0, 129)),
    output0=Output("output0", "TENSOR_BOOL8", "{3}"),
    input0_data=[129, 130, 131], # effectively 1, 2, 3
    input1_data=[131],           # effectively 2
    output_data=[True, True, False],
    do_variations=False,
)

test(
    name="quantized_overflow_second_input_if_requantized",
    input0=Input("input0", ("TENSOR_QUANT8_ASYMM", [1], 1.64771, 31)),
    input1=Input("input1", ("TENSOR_QUANT8_ASYMM", [1], 1.49725, 240)),
    output0=Output("output0", "TENSOR_BOOL8", "{1}"),
    input0_data=[0],
    input1_data=[200],
    output_data=[False],
    do_variations=False,
)

test(
    name="quantized_overflow_first_input_if_requantized",
    input0=Input("input0", ("TENSOR_QUANT8_ASYMM", [1], 1.49725, 240)),
    input1=Input("input1", ("TENSOR_QUANT8_ASYMM", [1], 1.64771, 31)),
    output0=Output("output0", "TENSOR_BOOL8", "{1}"),
    input0_data=[200],
    input1_data=[0],
    output_data=[True],
    do_variations=False,
)

test(
    name="boolean",
    input0=Input("input0", "TENSOR_BOOL8", "{4}"),
    input1=Input("input1", "TENSOR_BOOL8", "{4}"),
    output0=Output("output0", "TENSOR_BOOL8", "{4}"),
    input0_data=[False, True, False, True],
    input1_data=[False, False, True, True],
    output_data=[True, False, True, True],
    do_variations=False,
)
