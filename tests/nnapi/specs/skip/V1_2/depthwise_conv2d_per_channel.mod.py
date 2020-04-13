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

# TEST 1: Same scales, zeroPoint = 0
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 2}, 0.5f, 0")
f1 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{1, 2, 2, 2}, 0.0f, 0",
               [2, 4,  2, 0,  2, 2,  2, 0],
               extraParams = SymmPerChannelQuantParams(channelDim=3, scales=[0.5, 0.5]))
b1 = Parameter("op3", "TENSOR_INT32", "{2}", [0, 0])
o1 = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 1, 1, 2}, 1.f, 0")
Model("same").Operation("DEPTHWISE_CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 1, 0).To(o1)

# Instantiate an example
Example({
    i1: [4, 16, 4, 32, 4, 64, 4, 128],
    o1: [8, 48],
}).AddInput(f1, b1)


# TEST 2: Different scales, zeroPoint=128
i2 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 3, 2}, 0.5f, 128")
f2 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{1, 2, 2, 4}, 0.0f, 0",
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               extraParams = SymmPerChannelQuantParams(channelDim=3, scales=[1.0, 0.5, 1.0, 0.5]))
b2 = Parameter("op3", "TENSOR_INT32", "{4}", [4, 4, 4, 4])
o2 = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 4}, 1.f, 128")
Model("different").Operation("DEPTHWISE_CONV_2D", i2, f2, b2, 0, 0, 0, 0, 1, 1, 2, 0).To(o2)

# Instantiate an example
Example({
    i2: [129, 130] * 9,
    o2: [132, 130, 134, 131, 132, 130, 134, 131,
         132, 130, 134, 131, 132, 130, 134, 131],
}).AddInput(f2, b2)


layout = BoolScalar("layout", False) # NHWC

# TEST 3: With layout param
i3 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 3, 2}, 0.5f, 128")
f3 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{1, 2, 2, 4}, 0.0f, 0",
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               extraParams = SymmPerChannelQuantParams(channelDim=3, scales=[1.0, 0.5, 1.0, 0.5]))
b3 = Parameter("op3", "TENSOR_INT32", "{4}", [4, 4, 4, 4])
o3 = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 4}, 1.f, 128")
Model("layout").Operation("DEPTHWISE_CONV_2D", i3, f3, b3, 0, 0, 0, 0, 1, 1, 2, 0, layout).To(o3)

# Instantiate an example
Example({
    i3: [129, 130] * 9,
    o3: [132, 130, 134, 131, 132, 130, 134, 131,
         132, 130, 134, 131, 132, 130, 134, 131],
}).AddNchw(i3, o3, layout).AddInput(f3, b3)
