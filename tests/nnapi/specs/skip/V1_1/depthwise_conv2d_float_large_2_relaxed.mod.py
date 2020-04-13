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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 4}") # depth_in = 4
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0, 10, 100, .25, 1, 20, 100, .25, 0, 30, 100, .25, 1, 40, 100]) # depth_out = 4
b1 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [6000, 7000, 8000, 9000]) # depth_out = 4
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 1)
output = Output("op4", "TENSOR_FLOAT32", "{1, 1, 1, 4}")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {
    i1: [ # input 0
     10, 21, 10, 0,
     10, 22, 20, 0,
     10, 23, 30, 0,
     10, 24, 40, 0],
  }
# (i1 (conv) f1) + b1
output0 = {output: # output 0
           [6010, 7046, 11000, 9000]}

# Instantiate an example
Example((input0, output0))
