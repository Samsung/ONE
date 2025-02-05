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

# TEST 1: HEATMAP_MAX_KEYPOINT_1
heatmap1 = Input("heatmap", "TENSOR_FLOAT32", "{6, 4, 4, 1}")
boxes1 = Input("boxes", "TENSOR_FLOAT32", "{6, 4}")
score1 = Output("score", "TENSOR_FLOAT32", "{6, 1}")
keypoint1 = Output("keypoint", "TENSOR_FLOAT32", "{6, 1, 2}")
Model().Operation("HEATMAP_MAX_KEYPOINT", heatmap1, boxes1, layout).To(score1, keypoint1)

# Instantiate an example
Example({
    heatmap1: [
        -10, -1,  4, -5, # batch0
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5,
        -10, -1,  4, -5, # batch1 - test mirror bottom
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2, 10, -3,  5,
        -10, -1,  4, -5, # batch2 - test mirror left
         -8, -2,  4,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5,
        -10, -1,  4, 10, # batch3 - test mirror top right
         -8, -2,  4,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5,
        -10,-56,  4, -5, # batch4 - test out of range delta
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5,
        -10,-57.827329175, 4, -5, # batch5 - test detA = 0
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5
    ],
    boxes1: [
        5, 2, 10, 20,
        1, 7, 30, 10,
        8, 3, 15, 13,
        6, 5, 19, 12,
        5, 2, 10, 20,
        5, 2, 10, 20
    ],
    score1: [
        9.071493,
        10.00500,
        7.187500,
        10.00000,
        10.689667,
        9.000000
    ],
    keypoint1: [
        8.224462, 8.537316,
        11.73000, 9.625000,
        8.875000, 9.562500,
        17.37500, 5.875000,
        9.569672, 2.000000,
        8.125000, 8.750000
    ]
}).AddNchw(heatmap1, layout).AddVariations("relaxed", "float16")


# TEST 2: HEATMAP_MAX_KEYPOINT_2
heatmap2 = Input("heatmap", "TENSOR_FLOAT32", "{2, 4, 4, 4}")
boxes2 = Input("boxes", "TENSOR_FLOAT32", "{2, 4}")
score2 = Output("score", "TENSOR_FLOAT32", "{2, 4}")
keypoint2 = Output("keypoint", "TENSOR_FLOAT32", "{2, 4, 2}")
Model().Operation("HEATMAP_MAX_KEYPOINT", heatmap2, boxes2, layout).To(score2, keypoint2)

quant8 = DataTypeConverter().Identify({
    heatmap2: ("TENSOR_QUANT8_ASYMM", 0.01, 128),
    boxes2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    score2: ("TENSOR_QUANT8_ASYMM", 0.01, 0),
    keypoint2: ("TENSOR_QUANT16_ASYMM", 0.125, 0)
})

# Instantiate an example
Example({
    heatmap2: [
        0.19, 0.61, 0.49, 0.01, 0.98, 0.65, 0.64, 0.70, 0.76, 0.55,
        0.83, 0.19, 0.46, 0.03, 0.67, 0.71, 0.17, 0.23, 0.89, 0.08,
        0.96, 0.65, 0.52, 0.40, 0.36, 0.80, 0.55, 0.89, 0.58, 0.29,
        0.27, 0.69, 0.66, 0.06, 0.51, 0.26, 0.96, 0.38, 0.41, 0.89,
        0.88, 0.46, 0.96, 0.73, 0.54, 0.64, 0.84, 0.74, 0.51, 0.41,
        0.13, 0.19, 0.52, 0.21, 0.50, 0.75, 0.89, 0.89, 0.20, 0.58,
        0.70, 0.13, 0.29, 0.39,
        0.91, 0.06, 0.93, 0.34, 0.80, 0.87, 0.59, 0.67, 0.57, 0.85,
        0.24, 0.25, 0.76, 0.34, 0.37, 0.11, 0.00, 0.29, 0.30, 0.77,
        0.34, 0.57, 0.48, 0.76, 0.93, 0.18, 0.64, 0.12, 0.67, 0.47,
        0.56, 0.50, 0.48, 0.99, 0.46, 0.66, 0.98, 0.06, 0.10, 0.66,
        0.66, 0.91, 0.67, 0.23, 0.40, 0.37, 0.17, 0.35, 0.48, 0.98,
        0.47, 0.49, 0.56, 0.18, 0.75, 0.29, 0.04, 0.23, 0.42, 0.55,
        0.38, 0.07, 0.71, 0.80
    ],
    boxes2: [
        5, 2, 10, 20,
        1, 7, 30, 10
    ],
    score2: [
         1.020210,  0.890556,  1.007110,  0.945129,
         0.987798,  1.073820,  0.930000,  0.800000
    ],
    keypoint2: [
         7.227723,  4.250000,
         8.090278, 17.750000,
         8.523379, 12.589181,
         8.365580, 10.122508,
        12.431603,  8.934225,
         4.625000,  9.239437,
         4.625000,  7.375000,
        26.375000,  9.625000
    ]
}).AddNchw(heatmap2, layout).AddVariations("relaxed", "float16", quant8)


# TEST 3: HEATMAP_MAX_KEYPOINT_3
heatmap3 = Input("heatmap", "TENSOR_FLOAT32", "{5, 4, 4, 1}")
boxes3 = Input("boxes", "TENSOR_FLOAT32", "{5, 4}")
score3 = Output("score", "TENSOR_FLOAT32", "{5, 1}")
keypoint3 = Output("keypoint", "TENSOR_FLOAT32", "{5, 1, 2}")
Model().Operation("HEATMAP_MAX_KEYPOINT", heatmap3, boxes3, layout).To(score3, keypoint3)

quant8 = DataTypeConverter().Identify({
    heatmap3: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    boxes3: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    score3: ("TENSOR_QUANT8_ASYMM", 0.1, 10),
    keypoint3: ("TENSOR_QUANT16_ASYMM", 0.125, 0)
})

# Instantiate an example
Example({
    heatmap3: [
        -10, -1,  4, -5, # batch0
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5,
        -10, -1,  4, -5, # batch1 - test mirror bottom
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2, 10, -3,  5,
        -10, -1,  4, -5, # batch2 - test mirror left
         -8, -2,  4,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5,
        -10, -1,  4, 10, # batch3 - test mirror top right
         -8, -2,  4,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5,
        -10,-56,  4, -5, # batch4 - test out of range delta
         -8, -2,  9,  1,
          7, -2,  3, -7,
         -2,  2, -3,  5
    ],
    boxes3: [
        5, 2, 10, 20,
        1, 7, 30, 10,
        8, 3, 15, 13,
        6, 5, 19, 12,
        5, 2, 10, 20
    ],
    score3: [
        9.071493,
        10.00500,
        7.187500,
        10.00000,
        10.689667
    ],
    keypoint3: [
        8.224462, 8.537316,
        11.73000, 9.625000,
        8.875000, 9.562500,
        17.37500, 5.875000,
        9.569672, 2.000000
    ]
}).AddNchw(heatmap3, layout).AddVariations(quant8, includeDefault=False)
