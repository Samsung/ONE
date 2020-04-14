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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 4, 1}") # input 0
cons2 = Int32Scalar("cons2", 2)
pad_same = Int32Scalar("pad_same", 1)
act_none = Int32Scalar("act_none", 0)
i3 = Output("op3", "TENSOR_FLOAT32", "{1, 1, 2, 1}") # output 0
model = model.Operation("AVERAGE_POOL_2D", i1, pad_same, cons2, cons2, cons2, cons2, act_none).To(i3)
# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 6, 2, 4, 3, 2, 10, 7]}
output0 = {i3: # output 0
          [2.75, 5.75]}
# Instantiate an example
Example((input0, output0))
