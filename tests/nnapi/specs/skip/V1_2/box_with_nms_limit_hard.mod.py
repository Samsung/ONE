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

# TEST 1: BOX_WITH_NMS_LIMIT, score_threshold = 0.3, nms_threshold = 0.4, max_detections = -1
i1 = Input("scores", "TENSOR_FLOAT32", "{19, 3}") # scores
i2 = Input("roi", "TENSOR_FLOAT32", "{19, 12}") # roi
i3 = Input("batchSplit", "TENSOR_INT32", "{19}") # batchSplit

o1 = Output("scoresOut", "TENSOR_FLOAT32", "{12}") # scores out
o2 = Output("roiOut", "TENSOR_FLOAT32", "{12, 4}") # roi out
o3 = Output("classesOut", "TENSOR_INT32", "{12}") # classes out
o4 = Output("batchSplitOut", "TENSOR_INT32", "{12}") # batch split out
model = Model().Operation("BOX_WITH_NMS_LIMIT", i1, i2, i3, 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, o2, o3, o4)

quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.01, 0),
    i2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.01, 0),
    o2: ("TENSOR_QUANT16_ASYMM", 0.125, 0)
})

input0 = {
    i1: [   # scores
        0.90, 0.95, 0.75,
        0.80, 0.70, 0.85,
        0.60, 0.90, 0.95,
        0.90, 0.65, 0.90,
        0.80, 0.85, 0.80,
        0.60, 0.60, 0.20,
        0.60, 0.80, 0.40,
        0.90, 0.55, 0.60,
        0.90, 0.75, 0.70,
        0.80, 0.70, 0.85,
        0.90, 0.95, 0.75,
        0.80, 0.85, 0.80,
        0.60, 0.90, 0.95,
        0.60, 0.60, 0.20,
        0.50, 0.90, 0.80,
        0.90, 0.75, 0.70,
        0.90, 0.65, 0.90,
        0.90, 0.55, 0.60,
        0.60, 0.80, 0.40
    ],
    i2: [   # roi
        1, 1, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10,
        2, 2, 11, 11, 1, 1, 11, 11, 1, 1, 11, 11,
        3, 3, 12, 12, 2, 2, 12, 12, 2, 2, 12, 12,
        4, 4, 13, 13, 3, 3, 13, 13, 3, 3, 13, 13,
        5, 5, 14, 14, 4, 4, 14, 14, 4, 4, 14, 14,
        6, 6, 15, 15, 5, 5, 15, 15, 5, 5, 15, 15,
        7, 7, 16, 16, 6, 6, 16, 16, 6, 6, 16, 16,
        8, 8, 17, 17, 7, 7, 17, 17, 7, 7, 17, 17,
        9, 9, 18, 18, 8, 8, 18, 18, 8, 8, 18, 18,
        2, 2, 11, 11, 2, 2, 12, 12, 2, 2, 12, 12,
        1, 1, 10, 10, 1, 1, 11, 11, 1, 1, 11, 11,
        5, 5, 14, 14, 5, 5, 15, 15, 5, 5, 15, 15,
        3, 3, 12, 12, 3, 3, 13, 13, 3, 3, 13, 13,
        6, 6, 15, 15, 6, 6, 16, 16, 6, 6, 16, 16,
        0, 0, 1,  1,  0, 0, 2,  2,  0, 0, 2,  2,
        9, 9, 18, 18, 9, 9, 19, 19, 9, 9, 19, 19,
        4, 4, 13, 13, 4, 4, 14, 14, 4, 4, 14, 14,
        8, 8, 17, 17, 8, 8, 18, 18, 8, 8, 18, 18,
        7, 7, 16, 16, 7, 7, 17, 17, 7, 7, 17, 17
    ],
    i3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # batch split
}

output0 = {
    o1: [0.95, 0.85, 0.75, 0.95, 0.7, 0.95, 0.9, 0.85, 0.75, 0.95, 0.8, 0.7],
    o2: [
        0, 0, 10, 10,
        4, 4, 14, 14,
        8, 8, 18, 18,
        2, 2, 12, 12,
        8, 8, 18, 18,
        1, 1, 11, 11,
        0, 0,  2,  2,
        5, 5, 15, 15,
        9, 9, 19, 19,
        3, 3, 13, 13,
        0, 0,  2,  2,
        9, 9, 19, 19
    ],
    o3: [1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2],
    o4: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
}

Example((input0, output0)).AddVariations("relaxed", "float16", quant8)


# TEST 2: BOX_WITH_NMS_LIMIT, score_threshold = 0.3, nms_threshold = 0.4, max_detections = 5
i1 = Input("scores", "TENSOR_FLOAT32", "{19, 3}") # scores
i2 = Input("roi", "TENSOR_FLOAT32", "{19, 12}") # roi
i3 = Input("batchSplit", "TENSOR_INT32", "{19}") # batchSplit

o1 = Output("scoresOut", "TENSOR_FLOAT32", "{10}") # scores out
o2 = Output("roiOut", "TENSOR_FLOAT32", "{10, 4}") # roi out
o3 = Output("classesOut", "TENSOR_INT32", "{10}") # classes out
o4 = Output("batchSplitOut", "TENSOR_INT32", "{10}") # batch split out
model = Model().Operation("BOX_WITH_NMS_LIMIT", i1, i2, i3, 0.3, 5, 0, 0.4, 0.5, 0.3).To(o1, o2, o3, o4)

quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.01, 128),
    i2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.01, 128),
    o2: ("TENSOR_QUANT16_ASYMM", 0.125, 0)
})

input0 = {
    i1: [   # scores
        0.90, 0.95, 0.75,
        0.80, 0.70, 0.85,
        0.60, 0.90, 0.95,
        0.90, 0.65, 0.90,
        0.80, 0.85, 0.80,
        0.60, 0.60, 0.20,
        0.60, 0.80, 0.40,
        0.90, 0.55, 0.60,
        0.90, 0.75, 0.70,
        0.80, 0.70, 0.85,
        0.90, 0.95, 0.75,
        0.80, 0.85, 0.80,
        0.60, 0.90, 0.95,
        0.60, 0.60, 0.20,
        0.50, 0.90, 0.80,
        0.90, 0.75, 0.70,
        0.90, 0.65, 0.90,
        0.90, 0.55, 0.60,
        0.60, 0.80, 0.40
    ],
    i2: [   # roi
        1, 1, 10, 10, 0, 0, 10, 10, 0, 0, 10, 10,
        2, 2, 11, 11, 1, 1, 11, 11, 1, 1, 11, 11,
        3, 3, 12, 12, 2, 2, 12, 12, 2, 2, 12, 12,
        4, 4, 13, 13, 3, 3, 13, 13, 3, 3, 13, 13,
        5, 5, 14, 14, 4, 4, 14, 14, 4, 4, 14, 14,
        6, 6, 15, 15, 5, 5, 15, 15, 5, 5, 15, 15,
        7, 7, 16, 16, 6, 6, 16, 16, 6, 6, 16, 16,
        8, 8, 17, 17, 7, 7, 17, 17, 7, 7, 17, 17,
        9, 9, 18, 18, 8, 8, 18, 18, 8, 8, 18, 18,
        2, 2, 11, 11, 2, 2, 12, 12, 2, 2, 12, 12,
        1, 1, 10, 10, 1, 1, 11, 11, 1, 1, 11, 11,
        5, 5, 14, 14, 5, 5, 15, 15, 5, 5, 15, 15,
        3, 3, 12, 12, 3, 3, 13, 13, 3, 3, 13, 13,
        6, 6, 15, 15, 6, 6, 16, 16, 6, 6, 16, 16,
        0, 0, 1,  1,  0, 0, 2,  2,  0, 0, 2,  2,
        9, 9, 18, 18, 9, 9, 19, 19, 9, 9, 19, 19,
        4, 4, 13, 13, 4, 4, 14, 14, 4, 4, 14, 14,
        8, 8, 17, 17, 8, 8, 18, 18, 8, 8, 18, 18,
        7, 7, 16, 16, 7, 7, 17, 17, 7, 7, 17, 17
    ],
    i3: [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # batch split
}

output0 = {
    o1: [0.95, 0.85, 0.75, 0.95, 0.7, 0.95, 0.9, 0.85, 0.95, 0.8],
    o2: [
        0, 0, 10, 10,
        4, 4, 14, 14,
        8, 8, 18, 18,
        2, 2, 12, 12,
        8, 8, 18, 18,
        1, 1, 11, 11,
        0, 0,  2,  2,
        5, 5, 15, 15,
        3, 3, 13, 13,
        0, 0,  2,  2
    ],
    o3: [1, 1, 1, 2, 2, 1, 1, 1, 2, 2],
    o4: [1, 1, 1, 1, 1, 3, 3, 3, 3, 3],
}

Example((input0, output0)).AddVariations("relaxed", "float16", quant8)
