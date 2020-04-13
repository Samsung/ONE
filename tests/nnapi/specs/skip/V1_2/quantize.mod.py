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

import numpy as np

num_values = 300
values = list(np.linspace(-10, 10, num_values))

for input_type in ["TENSOR_FLOAT32", "TENSOR_FLOAT16"]:
  for scale, offset in [(1.0, 0),
                        (1.0, 1),
                        (0.01, 120),
                        (10.0, 120)]:
    input0 = Input("input0", input_type, "{%d}" % num_values)
    output0 = Output("output0", input_type, "{%d}" % num_values)

    model = Model().Operation("QUANTIZE", input0).To(output0)

    quantizeOutput = DataTypeConverter().Identify({
        output0: ["TENSOR_QUANT8_ASYMM", scale, offset],
    })

    Example({
        input0: values,
        output0: values,
    }).AddVariations(quantizeOutput, includeDefault=False)


# Zero-sized input

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
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# QUANTIZE op with numBatches = 0.
o3 = Output("out", "TENSOR_QUANT8_ASYMM", "{0, 2, 2, 1}, 0.1f, 128") # out
model = model.Operation("QUANTIZE", zero_sized).To(o3)

# Create test case with dummy values.
Example({
    i1: [1],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddVariations("relaxed", "float16")
