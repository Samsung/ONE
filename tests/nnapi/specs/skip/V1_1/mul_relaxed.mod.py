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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
i2 = Input("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
act = Int32Scalar("act", 0) # an int32_t scalar fuse_activation
i3 = Output("op3", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
model = model.Operation("MUL", i1, i2, act).To(i3)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [2, -4, 8, -16],
          i2: # input 1
          [32, -16, -8, 4]}

output0 = {i3: # output 0
           [64, 64, -64, -64]}

# Instantiate an example
Example((input0, output0))
