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

# TEST 1: ROI_ALIGN_1, outputShape = [2, 2], spatialScale = [0.5, 0.5], samplingRatio = [4, 4]
i1 = Input("in", "TENSOR_FLOAT32", "{1, 4, 4, 1}")
roi1 = Input("roi", "TENSOR_FLOAT32", "{4, 4}")
o1 = Output("out", "TENSOR_FLOAT32", "{4, 2, 2, 1}")
Model().Operation("ROI_ALIGN", i1, roi1, [0, 0, 0, 0], 2, 2, 2.0, 2.0, 4, 4, layout).To(o1)

quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    roi1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.0625, 128)
})

# Instantiate an example
Example({
    i1: [
        -10, -1,  4, -5,
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2, 10, -3,  5
    ],
    roi1: [
        2, 2, 4, 4,
        0, 0, 8, 8,
        2, 0, 4, 8,
        0, 2, 8, 4
    ],
    o1: [
        0.375, 5.125, -0.375, 2.875,
        -0.5, -0.3125, 3.1875, 1.125,
         0.25, 4.25, 4.875, 0.625,
        -0.1875, 1.125, 0.9375, -2.625
    ]
}).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 2: ROI_ALIGN_2, outputShape = [2, 3], spatialScale = [0.25, 0.25], samplingRatio = [4, 4]
i2 = Input("in", "TENSOR_FLOAT32", "{4, 4, 8, 2}")
roi2 = Input("roi", "TENSOR_FLOAT32", "{4, 4}")
o2 = Output("out", "TENSOR_FLOAT32", "{4, 2, 3, 2}")
Model().Operation("ROI_ALIGN", i2, roi2, [0, 0, 3, 3], 2, 3, 4.0, 4.0, 4, 4, layout).To(o2)

quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.04, 0),
    roi2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o2: ("TENSOR_QUANT8_ASYMM", 0.03125, 10)
})

# Instantiate an example
Example({
    i2: [
        8.84, 8.88, 7.41, 5.60, 9.95, 4.37, 0.10, 7.64, 6.50, 9.47,
        7.55, 3.00, 0.89, 3.01, 6.30, 4.40, 1.64, 6.74, 6.16, 8.60,
        5.85, 3.17, 7.12, 6.79, 5.77, 6.62, 5.13, 8.44, 5.08, 7.12,
        2.84, 1.19, 8.37, 0.90, 7.86, 9.69, 1.97, 1.31, 4.42, 9.89,
        0.18, 9.00, 9.30, 0.44, 5.05, 6.47, 1.09, 9.50, 1.30, 2.18,
        2.05, 7.74, 7.66, 0.65, 4.18, 7.14, 5.35, 7.90, 1.04, 1.47,
        9.01, 0.95, 4.07, 0.65,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00,
        5.47, 2.64, 0.86, 4.86, 2.38, 2.45, 8.77, 0.06, 3.60, 9.28,
        5.84, 8.97, 6.89, 1.43, 3.90, 5.91, 7.40, 9.25, 3.12, 4.92,
        1.87, 3.22, 9.50, 6.73, 2.07, 7.30, 3.07, 4.97, 0.24, 8.91,
        1.09, 0.27, 7.29, 6.94, 2.31, 6.88, 4.33, 1.37, 0.86, 0.46,
        6.07, 3.81, 0.86, 6.99, 4.36, 1.92, 8.19, 3.57, 7.90, 6.78,
        4.64, 6.82, 6.18, 9.63, 2.63, 2.33, 1.36, 2.70, 9.99, 9.85,
        8.06, 4.80, 7.80, 5.43
    ],
    roi2: [
        4, 4, 28, 12,
        4, 4, 32, 16,
        7, 1, 29, 15,   # test rounding
        1, 7,  9, 11    # test roi with shape smaller than output
    ],
    o2: [
        5.150000, 5.491250, 4.733750, 7.100000, 4.827500,
        5.843750, 4.721250, 4.797500, 3.750000, 6.592500,
        5.452500, 3.362500,
        4.899396, 5.861696, 4.941504, 5.979741, 3.182904,
        6.111551, 5.141833, 4.631891, 3.903325, 4.627793,
        5.537240, 1.356019,
        4.845915, 3.618338, 3.301958, 6.250566, 2.930461,
        4.269676, 3.642174, 4.201423, 5.008657, 5.735293,
        7.426004, 4.819665,
        4.518229, 6.887344, 2.952656, 5.565781, 3.952786,
        2.552812, 5.191667, 6.854167, 3.920000, 6.512500,
        4.886250, 5.497708
    ]
}).AddNchw(i2, o2, layout).AddVariations("relaxed", quant8, "float16")


# TEST 3: ROI_ALIGN_3, outputShape = [2, 3], spatialScale = [0.25, 0.25], samplingRatio = [0, 0]
i3 = Input("in", "TENSOR_FLOAT32", "{2, 4, 8, 2}")
roi3 = Input("roi", "TENSOR_FLOAT32", "{4, 4}")
o3 = Output("out", "TENSOR_FLOAT32", "{4, 2, 3, 2}")
Model().Operation("ROI_ALIGN", i3, roi3, [0, 0, 1, 1], 2, 3, 4.0, 4.0, 0, 0, layout).To(o3)

quant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 0.04, 0),
    roi3: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o3: ("TENSOR_QUANT8_ASYMM", 0.03125, 10)
})

# Instantiate an example
Example({
    i3: [
        8.84, 8.88, 7.41, 5.60, 9.95, 4.37, 0.10, 7.64, 6.50, 9.47,
        7.55, 3.00, 0.89, 3.01, 6.30, 4.40, 1.64, 6.74, 6.16, 8.60,
        5.85, 3.17, 7.12, 6.79, 5.77, 6.62, 5.13, 8.44, 5.08, 7.12,
        2.84, 1.19, 8.37, 0.90, 7.86, 9.69, 1.97, 1.31, 4.42, 9.89,
        0.18, 9.00, 9.30, 0.44, 5.05, 6.47, 1.09, 9.50, 1.30, 2.18,
        2.05, 7.74, 7.66, 0.65, 4.18, 7.14, 5.35, 7.90, 1.04, 1.47,
        9.01, 0.95, 4.07, 0.65,
        5.47, 2.64, 0.86, 4.86, 2.38, 2.45, 8.77, 0.06, 3.60, 9.28,
        5.84, 8.97, 6.89, 1.43, 3.90, 5.91, 7.40, 9.25, 3.12, 4.92,
        1.87, 3.22, 9.50, 6.73, 2.07, 7.30, 3.07, 4.97, 0.24, 8.91,
        1.09, 0.27, 7.29, 6.94, 2.31, 6.88, 4.33, 1.37, 0.86, 0.46,
        6.07, 3.81, 0.86, 6.99, 4.36, 1.92, 8.19, 3.57, 7.90, 6.78,
        4.64, 6.82, 6.18, 9.63, 2.63, 2.33, 1.36, 2.70, 9.99, 9.85,
        8.06, 4.80, 7.80, 5.43
    ],
    roi3: [
        4, 4, 28, 12,
        4, 4, 32, 16,
        7, 1, 29, 15,   # test rounding
        1, 7,  9, 11    # test roi with shape smaller than output
    ],
    o3: [
        5.150000, 5.491250, 4.733750, 7.100000, 4.827500,
        5.843750, 4.721250, 4.797500, 3.750000, 6.592500,
        5.452500, 3.362500,
        4.869884, 5.908148, 4.941701, 5.955718, 3.113403,
        6.341898, 5.156389, 4.604016, 3.881782, 4.616123,
        5.690694, 1.237153,
        5.028047, 3.560944, 3.157656, 6.395469, 2.896243,
        4.336576, 3.563021, 4.057767, 5.053437, 6.028906,
        7.396966, 4.668906,
        4.385000, 6.905000, 2.815000, 5.502500, 4.161667,
        1.829167, 5.191667, 6.854167, 3.920000, 6.512500,
        5.106667, 5.612500
    ]
}).AddNchw(i3, o3, layout).AddVariations("relaxed", quant8, "float16")


# TEST 4: ROI_ALIGN_4, outputShape = [2, 2], spatialScale = [0.5, 1.0], samplingRatio = [0, 4]
i4 = Input("in", "TENSOR_FLOAT32", "{4, 4, 4, 1}")
roi4 = Input("roi", "TENSOR_FLOAT32", "{5, 4}")
o4 = Output("out", "TENSOR_FLOAT32", "{5, 2, 2, 1}")
Model().Operation("ROI_ALIGN", i4, roi4, [2, 2, 2, 2, 2],  2, 2, 2.0, 1.0, 0, 4, layout).To(o4)

quant8 = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    roi4: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o4: ("TENSOR_QUANT8_ASYMM", 0.0625, 128)
})

# Instantiate an example
Example({
    i4: [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -10, -1,  4, -5,
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2, 10, -3,  5,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    roi4: [
        1, 2, 2, 4,
        0, 0, 4, 8,
        1, 0, 2, 8,
        0, 2, 4, 4,
        0, 0, 0, 0
    ],
    o4: [
        0.375, 5.125, -0.375, 2.875,
        -0.5, -0.3125, 3.1875, 1.125,
         0.25, 4.25, 4.875, 0.625,
        -0.1875, 1.125, 0.9375, -2.625,
        -7.4375, -3.3125, -6.8125, -3.4375
    ]
}).AddNchw(i4, o4, layout).AddVariations("relaxed", quant8, "float16")


# TEST 5: ROI_ALIGN_zero_sized

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3,  -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# ROI_ALIGN op with numRois = 0.
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Output("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

quant8 = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    zero_sized: ("TENSOR_QUANT8_ASYMM", 0.1, 128)
})

# Create test case with dummy values.
Example({
    i1: [0],
    o1: [0],
    o2: [0],
    zero_sized: [0],
}).AddNchw(i1, zero_sized, layout).AddVariations("relaxed", quant8, "float16")


# TEST 6: ROI_ALIGN_6, hanging issue
i4 = Input("in", "TENSOR_FLOAT32", "{1, 512, 8, 1}")
roi4 = Input("roi", "TENSOR_FLOAT32", "{1, 4}")
o4 = Output("out", "TENSOR_FLOAT32", "{1, 128, 4, 1}")
Model().Operation("ROI_ALIGN", i4, roi4, [0], 128, 4, 1.0, 64.0, 10, 10, layout).To(o4)

quant8 = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    roi4: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o4: ("TENSOR_QUANT8_ASYMM", 0.0625, 128)
})

# Instantiate an example
Example({
    i4: [0] * (512 * 8),
    roi4: [450, 500, 466, 508],
    o4: [0] * (128 * 4)
}).AddNchw(i4, o4, layout).AddVariations("relaxed", quant8, "float16")
