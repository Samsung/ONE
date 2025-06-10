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
i2 = Output("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
w = Int32Scalar("width", 3)
h = Int32Scalar("height", 3)
model = model.Operation("RESIZE_BILINEAR", i1, w, h).To(i2)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 1.0, 2.0, 2.0]}
output0 = {i2: # output 0
           [1.0, 1.0, 1.0,
            1.666666667, 1.666666667, 1.666666667,
            2.0, 2.0, 2.0]}

# Instantiate an example
Example((input0, output0))
