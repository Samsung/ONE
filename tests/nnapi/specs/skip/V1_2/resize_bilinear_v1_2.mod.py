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

# TEST 1: RESIZE_BILINEAR_NCHW_1, w = 3, h = 3
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
model_shape = Model("shape").Operation("RESIZE_BILINEAR", i1, 3, 3, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_BILINEAR", i1, 1.5, 1.5, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.01, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.01, 0)
})

test1 = {
    i1: [1.0, 1.0, 2.0, 2.0],
    o1: [1.0, 1.0, 1.0,
         1.666666667, 1.666666667, 1.666666667,
         2.0, 2.0, 2.0]
}

# Instantiate an example
Example(test1, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", "float16", quant8)
Example(test1, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", "float16", quant8)


# TEST 2: RESIZE_BILINEAR_NCHW_2, w = 3, h = 3
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 2}")
model_shape = Model("shape").Operation("RESIZE_BILINEAR", i2, 3, 3, layout).To(o2)
model_scale = Model("scale").Operation("RESIZE_BILINEAR", i2, 1.6, 1.6, layout).To(o2)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.25, 0),
    o2: ("TENSOR_QUANT8_ASYMM", 0.25, 0)
})

test2 = {
    i2: [3, 4, 6, 10, 9, 10, 12, 16],
    o2: [3, 4, 5, 8, 6, 10,
         7, 8, 9, 12, 10, 14,
         9, 10, 11, 14, 12, 16,]
}

# Instantiate an example
Example(test2, model=model_shape).AddNchw(i2, o2, layout).AddVariations("relaxed", "float16", quant8)
Example(test2, model=model_scale).AddNchw(i2, o2, layout).AddVariations("relaxed", "float16", quant8)


# TEST 3: RESIZE_BILINEAR, w = 3, h = 3
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
o3 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
model_shape = Model("shape").Operation("RESIZE_BILINEAR", i3, 3, 3).To(o3)
model_scale = Model("scale").Operation("RESIZE_BILINEAR", i3, 1.8, 1.8).To(o3)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 0.01, 0),
    o3: ("TENSOR_QUANT8_ASYMM", 0.01, 0)
})

test3 = {
    i3: [1.0, 1.0, 2.0, 2.0],
    o3: [1.0, 1.0, 1.0,
         1.666666667, 1.666666667, 1.666666667,
         2.0, 2.0, 2.0]
}

# Instantiate an example
Example(test3, model=model_shape).AddVariations("float16", quant8, includeDefault=False)
Example(test3, model=model_scale).AddVariations("float16", quant8, includeDefault=False)


# TEST 4: zero-sized input, resize by output shape

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3,  -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# RESIZE_BILINEAR op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 3, 3, 1}") # out
model = model.Operation("RESIZE_BILINEAR", zero_sized, 3, 3, layout).To(o3)

quant8 = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    zero_sized: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    o3: ("TENSOR_QUANT8_ASYMM", 0.1, 128)
})

# Create test case with dummy values.
Example({
    i1: [1],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddNchw(i1, zero_sized, o3, layout).AddVariations("relaxed", quant8, "float16")


# TEST 5: zero-sized input, resize by scale

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3,  -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# RESIZE_BILINEAR op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 3, 3, 1}") # out
model = model.Operation("RESIZE_BILINEAR", zero_sized, 1.6, 1.6, layout).To(o3)

quant8 = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    zero_sized: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    o3: ("TENSOR_QUANT8_ASYMM", 0.1, 128)
})

# Create test case with dummy values.
Example({
    i1: [1],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddNchw(i1, zero_sized, o3, layout).AddVariations("relaxed", quant8, "float16")
