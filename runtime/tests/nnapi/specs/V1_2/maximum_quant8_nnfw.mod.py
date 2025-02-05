#
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
# Copyright (C) 2020 The Android Open Source Project
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

i1 = Input("input0", "TENSOR_QUANT8_ASYMM", "{3, 1, 2}, 1.0f, 100")
i2 = Input("input1", "TENSOR_QUANT8_ASYMM", "{3, 1, 2}, 1.0f, 100")
i3 = Output("output0", "TENSOR_QUANT8_ASYMM", "{3, 1, 2}, 1.0f, 100")

model = Model().Operation("MAXIMUM", i1, i2).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [129, 127, 125, 149, 123, 124],
          i2: # input 1
          [125, 127, 129, 151, 121, 124]}

output0 = {i3: # output 0
           [129, 127, 129, 151, 123, 124]}

# Instantiate an example
Example((input0, output0))
