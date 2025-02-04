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

# TEST 1: FULLY_CONNECTED
model = Model()
in0 = Input("op1", "TENSOR_FLOAT32", "{3, 1}")
weights = Parameter("op2", "TENSOR_FLOAT32", "{1, 1}", [2])
bias = Parameter("b0", "TENSOR_FLOAT32", "{1}", [4])
out0 = Output("op3", "TENSOR_FLOAT32", "{3, 1}")
act = Int32Scalar("act", 0)
model = model.Operation("FULLY_CONNECTED", in0, weights, bias, act).To(out0)

quant8_mult_gt_1 = DataTypeConverter(name="quant8_mult_gt_1").Identify({
    in0: ("TENSOR_QUANT8_ASYMM", 0.5, 127),
    weights: ("TENSOR_QUANT8_ASYMM", 0.5, 120),
    bias: ("TENSOR_INT32", 0.25, 0),
    out0: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
})

# Example 1. Input in operand 0,
input0 = {in0: # input 0
          [2, 32, 16]}
output0 = {out0: # output 0
               [8, 68, 36]}

# Instantiate an example
Example((input0, output0)).AddVariations("relaxed", "float16", quant8_mult_gt_1)

# FULLY_CONNECTED of data type TENSOR_FLOAT32 is introduced in V1_0.
Example.SetVersion("V1_0", "fully_connected_v1_2")

# TEST 2: FULLY_CONNECTED, zero-sized input

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3,  -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
layout = BoolScalar("layout", False) # NHWC
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 3}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 3}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# FULLY_CONNECTED op with numBatches = 0.
w = Parameter("weights", "TENSOR_FLOAT32", "{1, 3}", [1, 2, 3]) # weights
b = Parameter("bias", "TENSOR_FLOAT32", "{1}", [1]) # bias
o3 = Output("out", "TENSOR_FLOAT32", "{0, 1}") # out
model = model.Operation("FULLY_CONNECTED", zero_sized, w, b, 0).To(o3)

quant8 = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    zero_sized: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    w: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    b: ("TENSOR_INT32", 0.01, 0),
    o3: ("TENSOR_QUANT8_ASYMM", 0.1, 128)
})

# Create test case with dummy values.
Example({
    i1: [1, 2, 3],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddNchw(i1, zero_sized, layout).AddVariations("relaxed", quant8, "float16")
