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
in0 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 4, 2}")
weights = Parameter("op2", "TENSOR_FLOAT32", "{1, 24}",
      [-0.25449711, 0, -2.1247749, 0, -1.143796, 0, -1.0299346, 0, -2.2373879, 0, -0.083096743, 0, -1.3230739, 0, 0.15294921, 0, -0.53045893, 0, -0.46075189, 0, -1.4482396, 0, -1.609534, 0])
bias = Parameter("b0", "TENSOR_FLOAT32", "{1}",
     [0.70098364,])
out0 = Output("op3", "TENSOR_FLOAT32", "{1, 1}")
act_relu = Int32Scalar("act_relu", 0)
model = model.Operation("FULLY_CONNECTED", in0, weights, bias, act_relu).To(out0)

# Example 1. Input in operand 0,
input0 = {in0: # input 0
          [1.4910057783, 3.4019672871, -0.0598693565, -0.0065411143, -0.6461477280, 1.9235717058, 1.0784962177, 0.1765922010, -2.2495496273, -1.6010370255, -2.4747757912, -0.3825767934, 2.3058984280, 0.7288306952, -0.8964791894, -2.7584488392, -0.2875919640, 0.1335377693, 1.8338065147, -2.6112849712, 0.9390821457, 1.9843852520, -1.2190681696, 1.0274435282, ]}
output0 = {out0: # output 0
           [2.0375289917]}

# Instantiate an example
Example((input0, output0))
