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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 3}") # input 0
filter_width = Int32Scalar("filter_width", 2)
filter_height = Int32Scalar("filter_height", 2)
stride_width = Int32Scalar("stride_width", 1)
stride_height = Int32Scalar("stride_height", 1)
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
i3 = Output("op3", "TENSOR_FLOAT32", "{1, 1, 1, 3}") # output 0
model = model.Operation("L2_POOL_2D", i1, pad0, pad0, pad0, pad0,
                        stride_width, stride_height,
                        filter_width, filter_height,
                        act).To(i3)
model = model.RelaxedExecution(True)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0,  2.0,  3.0,
           4.0,  5.0,  6.0,
           7.0,  8.0,  9.0,
           10.0, 11.0, 12.0]}
output0 = {i3: # output 0
          [6.442049503326416, # sqrt(166/4)
           7.3143692016601562, # sqrt(214/4)
           8.2158384323120117]} # sqrt(270/4)
# Instantiate an example
Example((input0, output0))
