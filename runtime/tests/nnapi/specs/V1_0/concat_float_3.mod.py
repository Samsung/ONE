#
# Copyright (C) 2017 The Android Open Source Project
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

# model
model = Model()

row = 212
col1 = 60
col2 = 30
output_col = col1 + col2

input1 = Input("input1", "TENSOR_FLOAT32", "{%d, %d}" % (row, col1)) # input tensor 1
input2 = Input("input2", "TENSOR_FLOAT32", "{%d, %d}" % (row, col2)) # input tensor 2
axis1 = Int32Scalar("axis1", 1)
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (row, output_col)) # output
model = model.Operation("CONCATENATION", input1, input2, axis1).To(output)

# Example 1.
input1_values = [x for x in range(row * col1)]
input2_values = [-x for x in range(row * col2)]
input0 = {input1: input1_values,
          input2: input2_values}

output_values = [x for x in range(row * output_col)]
for r in range(row):
  for c1 in range(col1):
    output_values[r * output_col + c1] = input1_values[r * col1 + c1]
  for c2 in range(col2):
    output_values[r * output_col + col1 + c2] = input2_values[r * col2 + c2]

output0 = {output: output_values}

# Instantiate an example
Example((input0, output0))
