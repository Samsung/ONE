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

# TEST 1: No layout param specified
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 1, 2}, 0.5f, 128")
f1 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{3, 1, 1, 2}, 0.0f, 0",
               [1, 2, 1, 2, 1, 2], extraParams = SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 0.75, 1.0]))
b1 = Parameter("op3", "TENSOR_INT32", "{3}", [4, 4, 4])
o1 = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 3, 1, 3}, 1.f, 128")
Model().Operation("CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 0).To(o1)

# Instantiate an example
Example({
    i1: [138, 138, 138, 138, 138, 138],
    o1: [137, 141, 145, 137, 141, 145, 137, 141, 145]
}).AddInput(f1, b1)

# TEST 2: layout param, NHWC/NCHW layouts
layout = BoolScalar("layout", False) # NHWC
i2 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 1, 2}, 0.5f, 128")
f2 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{3, 1, 1, 2}, 0.0f, 0",
               [1, 2, 1, 2, 1, 2], extraParams = SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 0.75, 1.0]))
b2 = Parameter("op3", "TENSOR_INT32", "{3}", [4, 4, 4])
o2 = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 3, 1, 3}, 1.f, 128")
Model("layouts").Operation("CONV_2D", i2, f2, b2, 0, 0, 0, 0, 1, 1, 0, layout).To(o2)

# Instantiate an example
Example({
    i2: [138, 108, 138, 108, 138, 108],
    o2: [121, 118, 115, 121, 118, 115, 121, 118, 115]
}).AddNchw(i2, o2, layout).AddInput(f2, b2)

# TEST 3: zero-sized input

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_QUANT8_ASYMM", "{1, 2}, 0.1f, 128", [137, 129]) # scores
p2 = Parameter("roi", "TENSOR_QUANT16_ASYMM", "{1, 8}, 0.125f, 0", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_QUANT8_ASYMM", "{0}, 0.1f, 128") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_QUANT16_ASYMM", "{0, 4}, 0.125f, 0") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_QUANT8_ASYMM", "{1, 1, 1, 2}, 0.5f, 128")
zero_sized = Internal("featureMap", "TENSOR_QUANT8_ASYMM", "{0, 2, 2, 2}, 0.5f, 128")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# CONV_2D op with numBatches = 0.
w = Parameter("weights", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{3, 1, 1, 2}, 0.0f, 0",
              [1, 2, 1, 2, 1, 2], extraParams = SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 0.75, 1.0]))
b = Parameter("bias", "TENSOR_INT32", "{3}", [4, 4, 4])
o3 = Output("out", "TENSOR_QUANT8_ASYMM", "{0, 2, 2, 3}, 1.f, 128") # out
model = model.Operation("CONV_2D", zero_sized, w, b, 0, 0, 0, 0, 1, 1, 0, layout).To(o3)

# Create test case with dummy values.
Example({
    i1: [130, 130],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddNchw(i1, zero_sized, o3, layout)
