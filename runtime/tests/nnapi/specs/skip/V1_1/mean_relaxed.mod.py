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

model = Model()
i1 = Input("input", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
axis = Parameter("axis", "TENSOR_INT32", "{1}", [2])
keepDims = Int32Scalar("keepDims", 0)
output = Output("output", "TENSOR_FLOAT32", "{1, 2, 1}")

model = model.Operation("MEAN", i1, axis, keepDims).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0,
           3.0, 4.0]}

output0 = {output: # output 0
          [1.5,
           3.5]}

# Instantiate an example
Example((input0, output0))
