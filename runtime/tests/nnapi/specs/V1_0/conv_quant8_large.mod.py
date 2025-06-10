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
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM", "{3, 1, 1, 3}, 0.5, 0", [1, 4, 7, 2, 5, 8, 3, 6, 9])
b1 = Parameter("op3", "TENSOR_INT32", "{3}, 0.25, 0", [0, 0, 0])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
output = Output("op4",  "TENSOR_QUANT8_ASYMM", "{1, 2, 3, 3}, 1.0, 0")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [  1,   2,   3,   4,   5,   6,   7,   8,   9,
            10,  11,  12,  13,  14,  15,  16,  17,  18]}

output0 = {output: # output 0
           [  8,   9,   11,
              17,  21,  24,
              26,  32,  38,
              35,  43,  51,
              44,  54,  65,
              53,  66,  78]
          }

# Instantiate an example
Example((input0, output0))
