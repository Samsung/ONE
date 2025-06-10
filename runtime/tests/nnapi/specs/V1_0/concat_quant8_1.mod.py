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
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{2, 3}, 0.5f, 0") # input 0
i2 = Input("op2", "TENSOR_QUANT8_ASYMM", "{2, 3}, 0.5f, 0") # input 1
axis1 = Int32Scalar("axis1", 1)
r = Output("result", "TENSOR_QUANT8_ASYMM", "{2, 6}, 0.5f, 0") # output
model = model.Operation("CONCATENATION", i1, i2, axis1).To(r)

# Example 1.
input0 = {i1: [1, 2, 3, 4, 5, 6],
          i2: [7, 8, 9, 10, 11, 12]}
output0 = {r: [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]}

# Instantiate an example
Example((input0, output0))
