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
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 3}") # input tensor 0
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 3}") # input tensor 1
axis0 = Int32Scalar("axis0", 0)
r = Output("result", "TENSOR_FLOAT32", "{4, 3}") # output
model = model.Operation("CONCATENATION", i1, i2, axis0).To(r)
model = model.RelaxedExecution(True)

# Example 1.
input0 = {i1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          i2: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}
output0 = {r: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}

# Instantiate an example
Example((input0, output0))
