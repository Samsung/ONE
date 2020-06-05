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

layout = BoolScalar("layout", False) # NHWC

# TEST 1: TRANSPOSE_CONV2D_LARGE, pad = same, stride = 32
i1 = Input("op1", "TENSOR_FLOAT32", "{25, 1, 1, 1}") # input 0
w1 = Parameter("op2", "TENSOR_FLOAT32", "{16, 1, 1, 1}", [1] * 16) # weight
b1 = Parameter("op3", "TENSOR_FLOAT32", "{16}", [0] * 16) # bias
s1 = Int32Vector("shape", [25, 32, 32, 16]) # output shape
act = Int32Scalar("act", 0) # act = none
o1 = Output("op4", "TENSOR_FLOAT32", "{25, 32, 32, 16}") # output
Model().Operation("TRANSPOSE_CONV_2D", i1, w1, b1, s1, 1, 32, 32, act, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    w1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    b1: ("TENSOR_INT32", 0.25, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.5, 0)
})

# Per-channel quantization
channelQuant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    w1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5] * 16)),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.125] * 16, hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM", 0.5, 80)
})

Example({
    i1: [1] * 25,
    o1: ([1] * 16 + [0] * (32 * 32 - 1) * 16) * 25
}).AddVariations(quant8, channelQuant8, includeDefault=False)

