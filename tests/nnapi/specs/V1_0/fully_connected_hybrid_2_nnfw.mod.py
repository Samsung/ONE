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
in0 = Input("op1", "TENSOR_FLOAT32", "{4, 1, 5, 1}")
weights = Parameter("op2", "TENSOR_QUANT8_SYMM", "{3, 10}, 0.5f, 0",
         [2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
          2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
          2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
bias = Parameter("b0", "TENSOR_FLOAT32", "{3}", [1, 2, 3])
out0 = Output("op3", "TENSOR_FLOAT32", "{2, 3}")
act_relu = Int32Scalar("act_relu", 1)
model = model.Operation("FULLY_CONNECTED", in0, weights, bias, act_relu).To(out0)

# Example 1. Input in operand 0,
input0 = {in0: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, -9, 12.7,
           1, 2, 3, 4, 5, 6, 7, -8, 9, -12.7]}
output0 = {out0: # output 0
           [251, 252, 253, 31, 32, 33]}

# Instantiate an example
Example((input0, output0))
