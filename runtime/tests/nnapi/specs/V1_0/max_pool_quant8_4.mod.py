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
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 2, 4, 1}, 0.0625f, 0") # input 0
cons2 = Int32Scalar("cons2", 2)
pad_same = Int32Scalar("pad_same", 1)
act_none = Int32Scalar("act_none", 0)
i3 = Output("op3", "TENSOR_QUANT8_ASYMM", "{1, 1, 2, 1}, 0.0625f, 0") # output 0
model = model.Operation("MAX_POOL_2D", i1, pad_same, cons2, cons2, cons2, cons2, act_none).To(i3)
# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 96, 32, 64, 48, 32, 160, 112]}
output0 = {i3: # output 0
          [96, 160]}
# Instantiate an example
Example((input0, output0))
