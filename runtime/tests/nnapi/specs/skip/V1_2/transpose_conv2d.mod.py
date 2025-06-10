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

layout = BoolScalar("layout", False) # NHWC

# TEST 1: TRANSPOSE_CONV2D, pad = valid, stride = 2
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
w1 = Parameter("op2", "TENSOR_FLOAT32", "{2, 3, 3, 1}", [1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18]) # weight
b1 = Parameter("op3", "TENSOR_FLOAT32", "{2}", [-1.5, -2]) # bias
s1 = Int32Vector("shape", [1, 5, 5, 2]) # output shape
act = Int32Scalar("act", 0) # act = none
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 5, 5, 2}") # output
Model().Operation("TRANSPOSE_CONV_2D", i1, w1, b1, s1, 2, 2, 2, act, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    w1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    b1: ("TENSOR_INT32", 0.25, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.5, 0)
})

quant8_mult_gt_1 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.5, 100),
    w1: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    b1: ("TENSOR_INT32", 0.25, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 80)
})

# Per-channel quantization
channelQuant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    w1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25, 0.5])),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.0625, 0.125], hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM", 0.5, 80)
})

channelQuant8_mult_gt_1 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    w1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25, 0.5])),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.0625, 0.125], hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 80)
})

Example({
    i1: [1, 2, 3, 4],
    o1: [-0.5,  0,  1.5,  2,   5.5,   8,  4.5,  6,  8.5, 10,
          5.5,  6,  7.5,  8,  23.5,  26, 16.5, 18, 20.5, 22,
         14.5, 18, 22.5, 26,  60.5,  70, 40.5, 46, 52.5, 58,
         19.5, 22, 25.5, 28,  59.5,  66, 34.5, 38, 42.5, 46,
         37.5, 40, 43.5, 46, 101.5, 108, 58.5, 62, 66.5, 70]
}).AddNchw(i1, o1, s1, layout).AddAllActivations(o1, act).AddVariations("relaxed", quant8, quant8_mult_gt_1, channelQuant8, channelQuant8_mult_gt_1, "float16").AddInput(w1, b1)


# TEST 2: TRANSPOSE_CONV2D_LARGE, pad = same, stride = 3, act = relu
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 1, 2, 1}") # input 0
w2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [9, 5, 6, 9, 8, 5, 3, 1, 4]) # weight
b2 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [-1000]) # bias
s2 = Int32Vector("shape", [1, 3, 4, 1]) # output shape
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 4, 1}") # output
Model().Operation("TRANSPOSE_CONV_2D", i2, w2, b2, s2, 1, 3, 3, 1, layout).To(o2)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 2.0, 0),
    w2: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    b2: ("TENSOR_INT32", 0.5, 0),
    o2: ("TENSOR_QUANT8_ASYMM", 20.0, 50)
})

# Per-channel quantization
channelQuant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 2.0, 0),
    w2: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25])),
    b2: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5], hide=True)),
    o2: ("TENSOR_QUANT8_ASYMM", 20.0, 50)
})

Example({
    i2: [300, 500],
    o2: [500.,  800.,  3500., 1500.,
         1400., 500.,  3500., 3000.,
         0.,    200.,  500.,  0.]
}).AddNchw(i2, o2, s2, layout).AddVariations("relaxed", quant8, channelQuant8, "float16").AddInput(w2, b2)


# TEST 3: TRANSPOSE_CONV2D_SAME, outputShape = [1, 4, 4, 1], pad = same, stride = 1, act = none
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 2}") # input 0
w3 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 2}", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) # weight
b3 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0]) # bias
s3 = Int32Vector("shape", [1, 4, 4, 1]) # output shape
o3 = Output("op4", "TENSOR_FLOAT32", "{1, 4, 4, 1}") # output
Model().Operation("TRANSPOSE_CONV_2D", i3, w3, b3, s3, 1, 1, 1, 0, layout).To(o3)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 0.5, 100),
    w3: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    b3: ("TENSOR_INT32", 0.25, 0),
    o3: ("TENSOR_QUANT8_ASYMM", 16.0, 0)
})

Example({
    i3: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    o3: [184,  412,  568,  528,
         678,  1347, 1689, 1434,
         1494, 2715, 3057, 2442,
         1968, 3352, 3652, 2760]
}).AddNchw(i3, o3, s3, layout).AddVariations("relaxed", quant8, "float16").AddInput(w3, b3)


# TEST 4: TRANSPOSE_CONV2D_VALID, outputShape = [1, 6, 6, 1], pad = valid, stride = 1, act = none
i4 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 2}") # input 0
w4 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 2}", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) # weight
b4 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0]) # bias
s4 = Int32Vector("shape", [1, 6, 6, 1]) # output shape
o4 = Output("op4", "TENSOR_FLOAT32", "{1, 6, 6, 1}") # output
Model().Operation("TRANSPOSE_CONV_2D", i4, w4, b4, s4, 2, 1, 1, 0, layout).To(o4)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM", 0.25, 10),
    w4: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    b4: ("TENSOR_INT32", 0.125, 0),
    o4: ("TENSOR_QUANT8_ASYMM", 32.0, 80)
})

Example({
    i4: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    o4: [5,    22,   59,   101,  114,  83,
         52,   184,  412,  568,  528,  344,
         237,  678,  1347, 1689, 1434, 879,
         597,  1494, 2715, 3057, 2442, 1431,
         856,  1968, 3352, 3652, 2760, 1548,
         689,  1534, 2543, 2729, 2010, 1103]
}).AddNchw(i4, o4, s4, layout).AddVariations("relaxed", quant8, "float16").AddInput(w4, b4)


# TEST 5: TRANSPOSE_CONV2D_EXPLICIT, pad = [1, 2, 2, 1], stride = 1, act = none
i5 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 2}") # input 0
w5 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 2}", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) # weight
b5 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0]) # bias
o5 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}") # output
Model().Operation("TRANSPOSE_CONV_2D", i5, w5, b5, 1, 2, 2, 1, 1, 1, 0, layout).To(o5)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i5: ("TENSOR_QUANT8_ASYMM", 0.5, 100),
    w5: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    b5: ("TENSOR_INT32", 0.125, 0),
    o5: ("TENSOR_QUANT8_ASYMM", 20.0, 50)
})

Example({
    i5: [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    o5: [678,  1347, 1689,
         1494, 2715, 3057,
         1968, 3352, 3652]
}).AddNchw(i5, o5, layout).AddVariations("relaxed", quant8, "float16").AddInput(w5, b5)


# TEST 6: zero-sized input, implicit padding

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# TRANSPOSE_CONV_2D op with numBatches = 0.
w = Parameter("weights", "TENSOR_FLOAT32", "{2, 3, 3, 1}", [1, 3, 5, 7, 9, 11, 9, 7, 5, 2, 4, 6, 8, 10, 12, 10, 8, 6]) # weight
b = Parameter("bias", "TENSOR_FLOAT32", "{2}", [-1.5, -2]) # bias
s = Int32Vector("shape", [0, 5, 5, 2]) # output shape
o3 = Output("out", "TENSOR_FLOAT32", "{0, 5, 5, 2}") # out
model = model.Operation("TRANSPOSE_CONV_2D", zero_sized, w, b, s, 2, 2, 2, 0, layout).To(o3)

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
    i1: [1],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddNchw(i1, zero_sized, o3, s, layout).AddVariations("relaxed", quant8, "float16")


# TEST 7: zero-sized input, explicit padding

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 4, 4, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 4, 4, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# TRANSPOSE_CONV_2D op with numBatches = 0.
w = Parameter("weights", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1, 3, 5, 7, 9, 11, 9, 7, 5]) # weight
b = Parameter("bias", "TENSOR_FLOAT32", "{1}", [-1.5]) # bias
o3 = Output("out", "TENSOR_FLOAT32", "{0, 3, 3, 1}") # out
model = model.Operation("TRANSPOSE_CONV_2D", zero_sized, w, b, 1, 2, 2, 1, 1, 1, 0, layout).To(o3)

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
    i1: [1],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddNchw(i1, zero_sized, o3, layout).AddVariations("relaxed", quant8, "float16")


# TEST 8: TRANSPOSE_CONV2D_SAME, outputShape = [1, 4, 4, 1], pad = same, stride = 2, act = none
i8 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
w8 = Parameter("op2", "TENSOR_FLOAT32", "{1, 1, 1, 1}", [2]) # weight
b8 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0]) # bias
s8 = Int32Vector("shape", [1, 4, 4, 1]) # output shape
o8 = Output("op4", "TENSOR_FLOAT32", "{1, 4, 4, 1}") # output
Model().Operation("TRANSPOSE_CONV_2D", i8, w8, b8, s8, 1, 2, 2, 0, layout).To(o8)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i8: ("TENSOR_QUANT8_ASYMM", 0.5, 100),
    w8: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    b8: ("TENSOR_INT32", 0.25, 0),
    o8: ("TENSOR_QUANT8_ASYMM", 16.0, 0)
})

Example({
    i8: [1,  2,  3,  4],
    o8: [2, 0, 4, 0, 0, 0, 0, 0, 6, 0, 8, 0, 0, 0, 0, 0]
}).AddNchw(i8, o8, s8, layout).AddVariations("relaxed", quant8, "float16").AddInput(w8, b8)
