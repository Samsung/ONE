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

i1 = Input("input", "TENSOR_FLOAT32", "{2, 5}") # batch = 2, depth = 5
beta = Float32Scalar("beta", 1.)
output = Output("output", "TENSOR_FLOAT32", "{2, 5}")

# model 1
model = model.Operation("SOFTMAX", i1, beta).To(output)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1:
          [1., 2., 3., 4., 5.,
           -1., -2., -3., -4., -5.]}

output0 = {output:
           [0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647,
            0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231]}

# Instantiate an example
Example((input0, output0))
