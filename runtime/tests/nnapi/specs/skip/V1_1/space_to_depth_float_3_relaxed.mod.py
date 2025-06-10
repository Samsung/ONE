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
i1 = Input("input", "TENSOR_FLOAT32", "{1, 4, 4, 2}")
block = Int32Scalar("block_size", 2)
output = Output("output", "TENSOR_FLOAT32", "{1, 2, 2, 8}")

model = model.Operation("SPACE_TO_DEPTH", i1, block).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [10,   20,  11,  21,  12,  22, 13,   23,
           14,   24,  15,  25,  16,  26, 17,   27,
           18,   28,  19,  29, 110, 210, 111, 211,
          112,  212, 113, 213, 114, 214, 115, 215]}

output0 = {output: # output 0
           [10,   20,  11,  21,  14,  24,  15,  25,
            12,   22,  13,  23,  16,  26,  17,  27,
            18,   28,  19,  29, 112, 212, 113, 213,
            110, 210, 111, 211, 114, 214, 115, 215]}

# Instantiate an example
Example((input0, output0))
