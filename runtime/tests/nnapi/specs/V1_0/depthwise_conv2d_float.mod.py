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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 2}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0, .2, 0, .25, 0, 0, .3, .25, 0, 0, 0, .25, .1, 0, 0])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 2)
output = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [10, 21, 10, 22, 10, 23,
           10, 24, 10, 25, 10, 26,
           10, 27, 10, 28, 10, 29]}
# (i1 (conv) f1) + b1
# filter usage:
#   in_ch1 * f_1  --> output_d1
#   in_ch1 * f_2  --> output_d2
#   in_ch2 * f_3  --> output_d3
#   in_ch3 * f_4  --> output_d4
output0 = {output: # output 0
           [11, 3, 7.2, 10.6,
            11, 3, 7.4, 10.9,
            11, 3, 7.8, 11.5,
            11, 3, 8.0, 11.8]}

# Instantiate an example
Example((input0, output0))
