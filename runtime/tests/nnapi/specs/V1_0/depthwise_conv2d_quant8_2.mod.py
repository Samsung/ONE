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
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 2, 2}, 0.5f, 127")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 4}, 0.5f, 127", [129, 131, 133, 135, 109, 147, 105, 151, 137, 139, 141, 143, 153, 99, 157, 95])
b1 = Parameter("op3", "TENSOR_INT32", "{4}, 0.25f, 0", [4, 8, 12, 16])
pad_valid = Int32Scalar("pad_valid", 2)
act_none = Int32Scalar("act_none", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 2)
output = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 2, 1, 4}, 1.f, 127")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad_valid,
                        stride, stride,
                        cm, act_none).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [129, 131, 141, 143,
           133, 135, 145, 147,
           137, 139, 149, 151]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [198, 93, 226, 107,
            218, 101, 254, 123]}

# Instantiate an example
Example((input0, output0))
