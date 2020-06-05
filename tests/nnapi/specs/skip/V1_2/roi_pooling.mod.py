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

# TEST 1: ROI_POOLING_1, outputShape = [2, 2], spatialScale = [0.5, 0.5]
i1 = Input("in", "TENSOR_FLOAT32", "{1, 4, 4, 1}")
roi1 = Input("roi", "TENSOR_FLOAT32", "{5, 4}")
o1 = Output("out", "TENSOR_FLOAT32", "{5, 2, 2, 1}")
Model().Operation("ROI_POOLING", i1, roi1, [0, 0, 0, 0, 0], 2, 2, 2.0, 2.0, layout).To(o1)

quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    roi1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 128)
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
        0, 0, 6, 6,
        2, 0, 4, 6,
        0, 2, 6, 4,
        8, 8, 8, 8  # empty region
    ],
    o1: [
        -2, 9, -2, 3,
        -1, 9, 10, 5,
        -1, 9, 10, 3,
        -2, 9,  7, 3,
         0, 0,  0, 0
    ]
}).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 2: ROI_POOLING_2, outputShape = [2, 3], spatialScale = 0.25
i2 = Input("in", "TENSOR_FLOAT32", "{4, 4, 8, 2}")
roi2 = Input("roi", "TENSOR_FLOAT32", "{4, 4}")
o2 = Output("out", "TENSOR_FLOAT32", "{4, 2, 3, 2}")
Model().Operation("ROI_POOLING", i2, roi2, [0, 0, 3, 3], 2, 3, 4.0, 4.0, layout).To(o2)

quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.04, 0),
    roi2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o2: ("TENSOR_QUANT8_ASYMM", 0.04, 0)
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
        4, 4, 24, 8,
        4, 4, 28, 12,
        7, 1, 25, 11,   # test rounding
        1, 7,  5, 11    # test roi with shape smaller than output
    ],
    o2: [
        6.16, 8.60, 7.12, 6.79, 5.13, 8.44, 7.86, 9.69, 4.42, 9.89, 9.30, 6.47,
        7.86, 9.89, 9.30, 9.89, 9.30, 9.50, 7.86, 9.89, 9.30, 9.89, 9.30, 9.50,
        9.50, 6.73, 9.50, 9.28, 6.89, 8.97, 6.18, 9.63, 9.99, 9.85, 9.99, 9.85,
        7.29, 6.94, 7.29, 6.94, 2.31, 6.88, 7.90, 6.78, 7.90, 6.82, 4.64, 6.82
    ]
}).AddNchw(i2, o2, layout).AddVariations("relaxed", quant8, "float16")


# TEST 3: ROI_POOLING_3, outputShape = [2, 2], spatialScale = [0.5, 1]
i3 = Input("in", "TENSOR_FLOAT32", "{4, 4, 4, 1}")
roi3 = Input("roi", "TENSOR_FLOAT32", "{5, 4}")
o3 = Output("out", "TENSOR_FLOAT32", "{5, 2, 2, 1}")
Model().Operation("ROI_POOLING", i3, roi3, [2, 2, 2, 2, 2], 2, 2, 2.0, 1.0, layout).To(o3)

quant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    roi3: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o3: ("TENSOR_QUANT8_ASYMM", 0.25, 128)
})

# Instantiate an example
Example({
    i3: [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -10, -1,  4, -5,
        -8, -2,  9,  1,
         7, -2,  3, -7,
        -2, 10, -3,  5,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    roi3: [
        1, 2, 2, 4,
        0, 0, 3, 6,
        1, 0, 2, 6,
        0, 2, 3, 4,
        0, 0, 0, 0
    ],
    o3: [
        -2, 9, -2, 3,
        -1, 9, 10, 5,
        -1, 9, 10, 3,
        -2, 9,  7, 3,
        -10, -10, -10, -10
    ]
}).AddNchw(i3, o3, layout).AddVariations("relaxed", quant8, "float16")
