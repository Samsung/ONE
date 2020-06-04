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
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 2, 3, 3}, 0.5, 0")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM", "{3, 1, 1, 3}, 0.5, 0")
b1 = Input("op3", "TENSOR_INT32", "{3}, 0.25, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
output = Output("op4",  "TENSOR_QUANT8_ASYMM", "{1, 2, 3, 3}, 1.0, 0")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [  1,   2,   3,   4,   5,   6,   7,   8,   9,
             10,  11,  12,  13,  14,  15,  16,  17,  18],
          f1:
          [ 10,  40,  70,
            20,  50,  80,
            30,  60,  90],
          b1:
          [0, 0, 0]}

output0 = {output: # output 0
           [  75,  90,  105,
              165, 203, 240,
              255, 255, 255,
              255, 255, 255,
              255, 255, 255,
              255, 255, 255]
          }

# Instantiate an example
Example((input0, output0))
