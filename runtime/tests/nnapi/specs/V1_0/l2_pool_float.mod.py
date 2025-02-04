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

model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
cons1 = Int32Scalar("cons1", 1)
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
i3 = Output("op3", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # output 0
model = model.Operation("L2_POOL_2D", i1, pad0, pad0, pad0, pad0, cons1, cons1, cons1, cons1, act).To(i3)
# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 3.0, 4.0]}
output0 = {i3: # output 0
          [1.0, 2.0, 3.0, 4.0]}
# Instantiate an example
Example((input0, output0))
