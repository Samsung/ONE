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
i1 = Input("input", "TENSOR_FLOAT32", "{4}")
begins = Parameter("begins", "TENSOR_INT32", "{1}", [1])
ends = Parameter("ends", "TENSOR_INT32", "{1}", [3])
strides = Parameter("strides", "TENSOR_INT32", "{1}", [1])
beginMask = Int32Scalar("beginMask", 0)
endMask = Int32Scalar("endMask", 0)
shrinkAxisMask = Int32Scalar("shrinkAxisMask", 0)

output = Output("output", "TENSOR_FLOAT32", "{2}")

model = model.Operation("STRIDED_SLICE", i1, begins, ends, strides, beginMask, endMask, shrinkAxisMask).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4]}

output0 = {output: # output 0
           [2, 3]}

# Instantiate an example
Example((input0, output0))
