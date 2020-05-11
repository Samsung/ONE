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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 4, 1}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1, 4, 7, 2, 5, 8, 3, 6, 9])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [-200])
pad_same = Int32Scalar("pad_same", 1)
act_relu = Int32Scalar("act_relu", 1)
stride = Int32Scalar("stride", 1)
output = Output("op4", "TENSOR_FLOAT32", "{1, 3, 4, 1}")

model = model.Operation("CONV_2D", i1, f1, b1, pad_same, stride, stride, act_relu).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

output0 = {output: # output 0
           [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0]}

# Instantiate an example
Example((input0, output0))
