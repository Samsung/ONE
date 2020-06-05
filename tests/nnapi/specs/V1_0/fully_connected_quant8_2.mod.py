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
in0 = Input("op1", "TENSOR_QUANT8_ASYMM", "{4, 1, 5, 1}, 0.5f, 127")
weights = Parameter("op2", "TENSOR_QUANT8_ASYMM", "{3, 10}, 0.5f, 127",
         [129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
          129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
          129, 131, 133, 135, 137, 139, 141, 143, 145, 147])
bias = Parameter("b0", "TENSOR_INT32", "{3}, 0.25f, 0", [4, 8, 12])
out0 = Output("op3", "TENSOR_QUANT8_ASYMM", "{2, 3}, 1.f, 127")
act_relu = Int32Scalar("act_relu", 1)
model = model.Operation("FULLY_CONNECTED", in0, weights, bias, act_relu).To(out0)

# Example 1. Input in operand 0,
input0 = {in0: # input 0
          [129, 131, 133, 135, 137, 139, 141, 143, 109, 107,
           129, 131, 133, 135, 137, 139, 141, 111, 145, 107]}
output0 = {out0: # output 0
           [151, 152, 153, 185, 186, 187]}

# Instantiate an example
Example((input0, output0))
