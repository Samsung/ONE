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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
f1 = Input("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
b1 = Input("op3", "TENSOR_FLOAT32", "{1}")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
# output dimension:
#     (i1.height - f1.height + 1) x (i1.width - f1.width + 1)
output = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 1}")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
          f1:
          [.25, .25, .25, .25],
          b1:
          [0]}

output0 = {output: # output 0
           [.875, .875, .875, .875]}

# Instantiate an example
Example((input0, output0))
