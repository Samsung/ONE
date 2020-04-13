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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 2, 2}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
pad_valid = Int32Scalar("pad_valid", 2)
act_none = Int32Scalar("act_none", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 2)
output = Output("op4", "TENSOR_FLOAT32", "{1, 2, 1, 4}")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad_valid,
                        stride, stride,
                        cm, act_none).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 7, 8,
           3, 4, 9, 10,
           5, 6, 11, 12]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [71, -34, 99, -20,
            91, -26, 127, -4]}

# Instantiate an example
Example((input0, output0))
