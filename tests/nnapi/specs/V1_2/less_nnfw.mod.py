#
# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 1}")
i2 = Input("op2", "TENSOR_FLOAT32", "{2}")
i3 = Output("op3", "TENSOR_BOOL8", "{2, 2}")
model = model.Operation("LESS", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [5, 10],
          i2: # input 1
          [10, 5]}

output0 = {i3: # output 0
           [True, False, False, False]}

# Instantiate an example
Example((input0, output0))
