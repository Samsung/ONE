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
  model = Model().Operation("REDUCE_ALL", input0, axes, keep_dims).To(output0)
  Example({
      input0: input_data,
      output0: output_data,
  }, model=model)

test(
    input0=Input("input0", "TENSOR_BOOL8", "{1}"),
    input_data=[False],
    axes=[0],
    keep_dims=True,
    output0=Output("output0", "TENSOR_BOOL8", "{1}"),
    output_data=[False],
)

test(
    input0=Input("input0", "TENSOR_BOOL8", "{2, 3, 2}"),
    input_data=[True, True, True, True, True, False,
                True, True, True, True, True, True],
    axes=[1, 0, -3, -3],
    keep_dims=False,
    output0=Output("output0", "TENSOR_BOOL8", "{2}"),
    output_data=[True, False],
)

test(
    input0=Input("input0", "TENSOR_BOOL8", "{2, 3, 2}"),
    input_data=[True, True, True, True, True, True,
                True, True, False, True, True, True],
    axes=[0, 2],
    keep_dims=True,
    output0=Output("output0", "TENSOR_BOOL8", "{1, 3, 1}"),
    output_data=[True, False, True],
)
