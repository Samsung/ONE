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

# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 1, 3, 3}") # a line of 3 pixels, 3 components/pixel
i2 = Input("op2", "TENSOR_INT32", "{1}") # another vector of 2 float32s
i3 = Output("op3", "TENSOR_FLOAT32", "{9}")
model = model.Operation("RESHAPE", i1, i2).To(i3)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3,
           4, 5, 6,
           7, 8, 9],
          i2: # input 1
          [-1]}

output0 = {i3: # output 0
           [1, 2, 3, 4, 5, 6, 7, 8, 9]}

# Instantiate an example
Example((input0, output0))
