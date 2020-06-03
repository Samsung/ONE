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
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 3, 1}, 0.5f, 0")
cons1 = Int32Scalar("cons1", 1)
pad0 = Int32Scalar("pad0", 0)
act2 = Int32Scalar("relu1_activitation", 2)
o = Output("op3", "TENSOR_QUANT8_ASYMM", "{1, 3, 3, 1}, 0.5f, 0")
model = model.Operation("AVERAGE_POOL_2D", i1, pad0, pad0, pad0, pad0, cons1, cons1, cons1, cons1, act2).To(o)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 3, 4, 5, 6, 7, 8]}

output0 = {o: # output 0
          [0, 1, 2, 2, 2, 2, 2, 2, 2]}

# Instantiate an example
Example((input0, output0))
