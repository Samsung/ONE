#
# Copyright (C) 2019 The Android Open Source Project
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

# Quantized SPACE_TO_BATCH_ND with non-zero zeroPoint is supported since 1.2.
# See http://b/132112227.

model = Model()
i1 = Input("input", "TENSOR_QUANT8_ASYMM", "{1, 5, 2, 1}, 1.0, 9")
block = Parameter("block_size", "TENSOR_INT32", "{2}", [3, 2])
paddings = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [1, 0, 2, 0])
output = Output("output", "TENSOR_QUANT8_ASYMM", "{6, 2, 2, 1}, 1.0, 9")

model = model.Operation("SPACE_TO_BATCH_ND", i1, block, paddings).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

output0 = {output: # output 0
           [9, 9, 9, 5, 9, 9, 9, 6, 9, 1, 9, 7,
            9, 2, 9, 8, 9, 3, 9, 9, 9, 4, 9, 10]}

# Instantiate an example
Example((input0, output0))
