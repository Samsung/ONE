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

# TEST 1: DETECTION_POSTPROCESSING
i1 = Input("scores", "TENSOR_FLOAT32", "{1, 6, 3}") # scores
i2 = Input("roi", "TENSOR_FLOAT32", "{1, 6, 4}") # roi
i3 = Input("anchors", "TENSOR_FLOAT32", "{6, 4}") # anchors

o1 = Output("scoresOut", "TENSOR_FLOAT32", "{1, 3}") # scores out
o2 = Output("roiOut", "TENSOR_FLOAT32", "{1, 3, 4}") # roi out
o3 = Output("classesOut", "TENSOR_INT32", "{1, 3}") # classes out
o4 = Output("detectionOut", "TENSOR_INT32", "{1}") # num detections out
Model("regular").Operation("DETECTION_POSTPROCESSING", i1, i2, i3, 10.0, 10.0, 5.0, 5.0, True, 3, 1, 1, 0.0, 0.5, False).To(o1, o2, o3, o4)

input0 = {
    i1: [   # class scores - two classes with background
        0., .9, .8,
        0., .75, .72,
        0., .6, .5,
        0., .93, .95,
        0., .5, .4,
        0., .3, .2
    ],
    i2: [   # six boxes in center-size encoding
        0.0, 0.0,  0.0, 0.0,  # box #1
        0.0, 1.0,  0.0, 0.0,  # box #2
        0.0, -1.0, 0.0, 0.0,  # box #3
        0.0, 0.0,  0.0, 0.0,  # box #4
        0.0, 1.0,  0.0, 0.0,  # box #5
        0.0, 0.0,  0.0, 0.0   # box #6
    ],
    i3: [   # six anchors in center-size encoding
        0.5, 0.5,   1.0, 1.0,  # anchor #1
        0.5, 0.5,   1.0, 1.0,  # anchor #2
        0.5, 0.5,   1.0, 1.0,  # anchor #3
        0.5, 10.5,  1.0, 1.0,  # anchor #4
        0.5, 10.5,  1.0, 1.0,  #  anchor #5
        0.5, 100.5, 1.0, 1.0   # anchor #6
    ]
}

output0 = {
    o1: [0.95, 0.93, 0.0],
    o2: [
        0.0, 10.0, 1.0, 11.0,
        0.0, 10.0, 1.0, 11.0,
        0.0, 0.0, 0.0, 0.0
    ],
    o3: [1, 0, 0],
    o4: [2],
}

Example((input0, output0)).AddVariations("relaxed", "float16")

# TEST 2: DETECTION_POSTPROCESSING
i1 = Input("scores", "TENSOR_FLOAT32", "{1, 6, 3}") # scores
i2 = Input("roi", "TENSOR_FLOAT32", "{1, 6, 4}") # roi
i3 = Input("anchors", "TENSOR_FLOAT32", "{6, 4}") # anchors

o1 = Output("scoresOut", "TENSOR_FLOAT32", "{1, 3}") # scores out
o2 = Output("roiOut", "TENSOR_FLOAT32", "{1, 3, 4}") # roi out
o3 = Output("classesOut", "TENSOR_INT32", "{1, 3}") # classes out
o4 = Output("detectionOut", "TENSOR_INT32", "{1}") # num detections out
Model().Operation("DETECTION_POSTPROCESSING", i1, i2, i3, 10.0, 10.0, 5.0, 5.0, False, 3, 1, 1, 0.0, 0.5, False).To(o1, o2, o3, o4)

input0 = {
    i1: [   # class scores - two classes with background
        0., .9, .8,
        0., .75, .72,
        0., .6, .5,
        0., .93, .95,
        0., .5, .4,
        0., .3, .2
    ],
    i2: [   # six boxes in center-size encoding
        0.0, 0.0,  0.0, 0.0,  # box #1
        0.0, 1.0,  0.0, 0.0,  # box #2
        0.0, -1.0, 0.0, 0.0,  # box #3
        0.0, 0.0,  0.0, 0.0,  # box #4
        0.0, 1.0,  0.0, 0.0,  # box #5
        0.0, 0.0,  0.0, 0.0   # box #6
    ],
    i3: [   # six anchors in center-size encoding
        0.5, 0.5,   1.0, 1.0,  # anchor #1
        0.5, 0.5,   1.0, 1.0,  # anchor #2
        0.5, 0.5,   1.0, 1.0,  # anchor #3
        0.5, 10.5,  1.0, 1.0,  # anchor #4
        0.5, 10.5,  1.0, 1.0,  #  anchor #5
        0.5, 100.5, 1.0, 1.0   # anchor #6
    ]
}

output0 = {
    o1: [0.95, 0.9, 0.3],
    o2: [
        0.0, 10.0, 1.0, 11.0,
        0.0, 0.0, 1.0, 1.0,
        0.0, 100.0, 1.0, 101.0
    ],
    o3: [1, 0, 0],
    o4: [3],
}

Example((input0, output0)).AddVariations("relaxed", "float16")

# TEST 3: DETECTION_POSTPROCESSING
i1 = Input("scores", "TENSOR_FLOAT32", "{1, 6, 3}") # scores
i2 = Input("roi", "TENSOR_FLOAT32", "{1, 6, 7}") # roi
i3 = Input("anchors", "TENSOR_FLOAT32", "{6, 4}") # anchors

o1 = Output("scoresOut", "TENSOR_FLOAT32", "{1, 3}") # scores out
o2 = Output("roiOut", "TENSOR_FLOAT32", "{1, 3, 4}") # roi out
o3 = Output("classesOut", "TENSOR_INT32", "{1, 3}") # classes out
o4 = Output("detectionOut", "TENSOR_INT32", "{1}") # num detections out
Model().Operation("DETECTION_POSTPROCESSING", i1, i2, i3, 10.0, 10.0, 5.0, 5.0, False, 3, 1, 1, 0.0, 0.5, False).To(o1, o2, o3, o4)

input0 = {
    i1: [   # class scores - two classes with background
        0., .9, .8,
        0., .75, .72,
        0., .6, .5,
        0., .93, .95,
        0., .5, .4,
        0., .3, .2
    ],
    i2: [   # six boxes in center-size encoding
        0.0, 0.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #1
        0.0, 1.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #2
        0.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0,  # box #3
        0.0, 0.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #4
        0.0, 1.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #5
        0.0, 0.0,  0.0, 0.0, 1.0, 2.0, 3.0   # box #6
    ],
    i3: [   # six anchors in center-size encoding
        0.5, 0.5,   1.0, 1.0,  # anchor #1
        0.5, 0.5,   1.0, 1.0,  # anchor #2
        0.5, 0.5,   1.0, 1.0,  # anchor #3
        0.5, 10.5,  1.0, 1.0,  # anchor #4
        0.5, 10.5,  1.0, 1.0,  #  anchor #5
        0.5, 100.5, 1.0, 1.0   # anchor #6
    ]
}

output0 = {
    o1: [0.95, 0.9, 0.3],
    o2: [
        0.0, 10.0, 1.0, 11.0,
        0.0, 0.0, 1.0, 1.0,
        0.0, 100.0, 1.0, 101.0
    ],
    o3: [1, 0, 0],
    o4: [3],
}

Example((input0, output0)).AddVariations("relaxed", "float16")

# TEST 4: DETECTION_POSTPROCESSING
i1 = Input("scores", "TENSOR_FLOAT32", "{1, 6, 3}") # scores
i2 = Input("roi", "TENSOR_FLOAT32", "{1, 6, 7}") # roi
i3 = Input("anchors", "TENSOR_FLOAT32", "{6, 4}") # anchors

o1 = Output("scoresOut", "TENSOR_FLOAT32", "{1, 3}") # scores out
o2 = Output("roiOut", "TENSOR_FLOAT32", "{1, 3, 4}") # roi out
o3 = Output("classesOut", "TENSOR_INT32", "{1, 3}") # classes out
o4 = Output("detectionOut", "TENSOR_INT32", "{1}") # num detections out
Model().Operation("DETECTION_POSTPROCESSING", i1, i2, i3, 10.0, 10.0, 5.0, 5.0, False, 3, 1, 1, 0.0, 0.5, True).To(o1, o2, o3, o4)

input0 = {
    i1: [   # class scores - two classes with background
        0., .9, .8,
        0., .75, .72,
        0., .6, .5,
        0., .93, .95,
        0., .5, .4,
        0., .3, .2
    ],
    i2: [   # six boxes in center-size encoding
        0.0, 0.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #1
        0.0, 1.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #2
        0.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0,  # box #3
        0.0, 0.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #4
        0.0, 1.0,  0.0, 0.0, 1.0, 2.0, 3.0,  # box #5
        0.0, 0.0,  0.0, 0.0, 1.0, 2.0, 3.0   # box #6
    ],
    i3: [   # six anchors in center-size encoding
        0.5, 0.5,   1.0, 1.0,  # anchor #1
        0.5, 0.5,   1.0, 1.0,  # anchor #2
        0.5, 0.5,   1.0, 1.0,  # anchor #3
        0.5, 10.5,  1.0, 1.0,  # anchor #4
        0.5, 10.5,  1.0, 1.0,  #  anchor #5
        0.5, 100.5, 1.0, 1.0   # anchor #6
    ]
}

output0 = {
    o1: [0.95, 0.9, 0.3],
    o2: [
        0.0, 10.0, 1.0, 11.0,
        0.0, 0.0, 1.0, 1.0,
        0.0, 100.0, 1.0, 101.0
    ],
    o3: [2, 1, 1],
    o4: [3],
}

Example((input0, output0)).AddVariations("relaxed", "float16")
