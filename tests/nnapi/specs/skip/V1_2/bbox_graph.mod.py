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

# Operation 1, GENERATE_PROPOSALS
scores = Input("scores", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
deltas = Input("deltas", "TENSOR_FLOAT32", "{1, 1, 1, 4}")
anchors = Input("anchors", "TENSOR_FLOAT32", "{1, 4}")
image = Input("imageInfo", "TENSOR_FLOAT32", "{1, 2}")
scoresOut_1 = Output("scores", "TENSOR_FLOAT32", "{0}")
roiOut_1 = Internal("roi", "TENSOR_FLOAT32", "{0, 4}")
batchOut_1 = Internal("batches", "TENSOR_INT32", "{0}")
model = Model("zero_sized").Operation("GENERATE_PROPOSALS", scores, deltas, anchors, image, 1.0, 1.0, -1, -1, 0.3, 10.0, layout).To(scoresOut_1, roiOut_1, batchOut_1)

# Operation 2, ROI_ALIGN
feature = Input("featureMap", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
featureOut_2 = Internal("scores", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", feature, roiOut_1, batchOut_1, 2, 2, 1.0, 1.0, 4, 4, layout).To(featureOut_2)

# Operation 3, FULLY_CONNECTED
weights_3 = Parameter("weights", "TENSOR_FLOAT32", "{8, 4}", [1] * 32)
bias_3 = Parameter("bias", "TENSOR_FLOAT32", "{8}", [1] * 8)
deltaOut_3 = Internal("delta", "TENSOR_FLOAT32", "{0, 8}")
model = model.Operation("FULLY_CONNECTED", featureOut_2, weights_3, bias_3, 0).To(deltaOut_3)

# Operation 4, FULLY_CONNECTED
weights_4 = Parameter("weights", "TENSOR_FLOAT32", "{2, 4}", [1] * 8)
bias_4 = Parameter("bias", "TENSOR_FLOAT32", "{2}", [1] * 2)
scoresOut_4 = Internal("scores", "TENSOR_FLOAT32", "{0, 2}")
model = model.Operation("FULLY_CONNECTED", featureOut_2, weights_4, bias_4, 0).To(scoresOut_4)

# Operation 5, AXIS_ALIGNED_BBOX_TRANSFORM
roiOut_5 = Internal("roi", "TENSOR_FLOAT32", "{0, 8}")
model = model.Operation("AXIS_ALIGNED_BBOX_TRANSFORM", roiOut_1, deltaOut_3, batchOut_1, image).To(roiOut_5)

# Operation 6, BOX_WITH_NMS_LIMIT
scoresOut_6 = Output("scores", "TENSOR_FLOAT32", "{0}")
roiOut_6 = Output("roi", "TENSOR_FLOAT32", "{0, 4}")
classOut_6 = Output("classes", "TENSOR_INT32", "{0}")
batchOut_6 = Output("batches", "TENSOR_INT32", "{0}")
model = model.Operation("BOX_WITH_NMS_LIMIT", scoresOut_4, roiOut_5, batchOut_1, 0.1, -1, 0, 0.3, 1.0, 0.1).To(scoresOut_6, roiOut_6, classOut_6, batchOut_6)

quant8 = DataTypeConverter().Identify({
    scores: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    deltas: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    anchors: ("TENSOR_QUANT16_SYMM", 0.125, 0),
    image: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    scoresOut_1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    roiOut_1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    feature: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    featureOut_2: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    weights_3: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    bias_3: ("TENSOR_INT32", 0.01, 0),
    deltaOut_3: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    weights_4: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    bias_4: ("TENSOR_INT32", 0.01, 0),
    scoresOut_4: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    roiOut_5: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    scoresOut_6: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    roiOut_6: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
})

Example({

    # Inputs that will lead to zero-sized output of GENERATE_PROPOSALS
    scores: [0.5],
    deltas: [0, 0, -10, -10],
    anchors: [0, 0, 10, 10],
    image: [32, 32],
    feature: [1],

    # Dummy outputs
    scoresOut_1: [0],
    scoresOut_6: [0],
    roiOut_6: [0],
    classOut_6: [0],
    batchOut_6: [0],

}).AddVariations("relaxed", "float16", quant8)
