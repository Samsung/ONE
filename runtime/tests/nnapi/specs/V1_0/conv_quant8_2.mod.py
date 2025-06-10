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

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 6, 1}, 0.5f, 127")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 0.5f, 127",
               [129, 131, 133, 135])
b1 = Parameter("op3", "TENSOR_INT32", "{1}, 0.25f, 0", [-4])
pad_valid = Int32Scalar("pad_valid", 2)
act_none = Int32Scalar("act_none", 0)
stride1 = Int32Scalar("stride1", 1)
stride3 = Int32Scalar("stride3", 3)

output = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 1.f, 127")

model = model.Operation("CONV_2D", i1, f1, b1, pad_valid, stride3,
                        stride1, act_none).To(output)

# Example 1. Input in operand 0,
input0 = {
    i1:  # input 0
        [133, 131, 129, 125, 123, 121,
         135, 133, 131, 123, 121, 119,
         137, 135, 133, 121, 119, 117]
}

output0 = {
    output:  # output 0
        [157, 103, 167, 93]
}

# Instantiate an example
Example((input0, output0))
