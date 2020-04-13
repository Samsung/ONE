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
  model = Model().Operation("LOGICAL_OR", input0, input1).To(output0)
  Example({
      input0: input0_data,
      input1: input1_data,
      output0: output_data,
  }, model=model, name=name)

test(
    name="simple",
    input0=Input("input0", "TENSOR_BOOL8", "{1, 1, 1, 4}"),
    input1=Input("input1", "TENSOR_BOOL8", "{1, 1, 1, 4}"),
    output0=Output("output0", "TENSOR_BOOL8", "{1, 1, 1, 4}"),
    input0_data=[True, False, False, True],
    input1_data=[True, False, True, False],
    output_data=[True, False, True, True],
)

test(
    name="broadcast",
    input0=Input("input0", "TENSOR_BOOL8", "{1, 1, 1, 4}"),
    input1=Input("input1", "TENSOR_BOOL8", "{1, 1}"),
    output0=Output("output0", "TENSOR_BOOL8", "{1, 1, 1, 4}"),
    input0_data=[True, False, False, True],
    input1_data=[False],
    output_data=[True, False, False, True],
)
