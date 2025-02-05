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

# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 0.5f, 0") # input 0
o = Output("op2", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 0.5f, 0") # output 0
model = model.Operation("RELU1", i1).To(o)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 3]}
output0 = {o: # output 0
          [0, 1, 2, 2]}

# Instantiate one example
Example((input0, output0))

# Example 2. Input in operand 0,
input1 = {i1: # input 0
          [4, 10, 100, 255]}
output1 = {o: # output 0
          [2, 2, 2, 2]}

# Instantiate another example
Example((input1, output1))
