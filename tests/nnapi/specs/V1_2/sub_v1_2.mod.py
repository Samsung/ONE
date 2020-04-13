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

import random

random.seed(0)

# FLOAT32 and FLOAT16
input0 = Input("input0", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
input1 = Input("input1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
activation = Int32Scalar("act", 0)
output0 = Output("output0", "TENSOR_FLOAT32",  "{1, 2, 2, 1}")

model = Model().Operation("SUB", input0, input1, activation).To(output0)

Example({
    input0: [2.0, -4.0, 8.0, -16.0],
    input1: [2.0, -2.0, -4.0, 4.0],
    output0: [0.0, -2.0, 12.0, -20.0],
}).AddVariations("float16").AddAllActivations(output0, activation)


# QUANT8_ASYMM
shape = "{2, 4, 16, 2}, 0.5, 0"
input0 = Input("input0", "TENSOR_QUANT8_ASYMM", shape)
input1 = Input("input1", "TENSOR_QUANT8_ASYMM", shape)
activation = 0
output0 = Output("output0", "TENSOR_QUANT8_ASYMM", shape)

model = Model("quant8").Operation("SUB", input0, input1, activation).To(output0)

input0_values = list(range(256))
input1_values = list(input0_values)
random.shuffle(input1_values)
output_values = [max(0, a - b) for a, b in zip(input0_values, input1_values)]

Example({
    input0: input0_values,
    input1: input1_values,
    output0: output_values,
})

# SUB of data type TENSOR_FLOAT32 is introduced in V1_1.
Example.SetVersion("V1_1", "sub_v1_2_none", "sub_v1_2_relu", "sub_v1_2_relu1", "sub_v1_2_relu6")


# SUB, zero-sized input

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
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 2}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 2}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# SUB op with numBatches = 0.
i2 = Parameter("op", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [1, 2, 3, 4]) # weights
o3 = Output("out", "TENSOR_FLOAT32", "{0, 2, 2, 2}") # out
model = model.Operation("SUB", zero_sized, i2, 0).To(o3)

quant8 = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    zero_sized: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    i2: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    o3: ("TENSOR_QUANT8_ASYMM", 0.1, 128)
})

# Create test case with dummy values.
Example({
    i1: [1, 2],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddVariations("relaxed", quant8, "float16")
