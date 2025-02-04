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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 3}")
i2 = Output("op2", "TENSOR_FLOAT32", "{1, 2, 2, 3}")

# Color wize (channel-wise) normalization
model = model.Operation("L2_NORMALIZATION", i1).To(i2)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0 - 4 color, 3 channels
          [0, 3,  4, # 5
           0, 5, 12, # 13,
           0, 8, 15, # 17,
           0, 7, 24]} # 25

output0 = {i2: # output 0
           [0, .6, .8,
            0, 0.38461539149284363, 0.92307698726654053,
            0, 0.47058823704719543, 0.88235294818878174,
            0, 0.28, 0.96]}

# Instantiate an example
Example((input0, output0))
