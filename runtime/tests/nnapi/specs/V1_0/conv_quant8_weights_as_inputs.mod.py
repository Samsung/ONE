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

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 3, 1}, 0.5f, 0")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 0.5f, 0")
b1 = Input("op3", "TENSOR_INT32", "{1}, 0.25f, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
# output dimension:
#     (i1.height - f1.height + 1) x (i1.width - f1.width + 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 1.f, 0")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [8, 8, 8, 8, 4, 8, 8, 8, 8],
          f1:
          [2, 2, 2, 2],
          b1:
          [4]}
# (i1 (conv) f1) + b1
output0 = {output: # output 0
           [15, 15, 15, 15]}

# Instantiate an example
Example((input0, output0))
