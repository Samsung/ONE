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
in0 = Input("op1", "TENSOR_FLOAT32", "{2, 2}")
weights = Parameter("op2", "TENSOR_FLOAT32", "{1, 2}", [2, 4])
bias = Parameter("b0", "TENSOR_FLOAT32", "{1}", [1])
out0 = Output("op3", "TENSOR_FLOAT32", "{2, 1}")
act = Int32Scalar("act", 0)
model = model.Operation("FULLY_CONNECTED", in0, weights, bias, act).To(out0)

# Example 1. Input in operand 0,
input0 = {in0: # input 0
          [1, 2, 2, 1]}
output0 = {out0: # output 0
               [11, 9]}

# Instantiate an example
Example((input0, output0))
