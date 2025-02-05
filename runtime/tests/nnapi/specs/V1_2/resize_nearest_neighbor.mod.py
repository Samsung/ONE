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

# TEST 1: RESIZE_NEAREST_NEIGHBOR_1, w = 1, h = 1
i1 = Input("in", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{1, 1, 1, 1}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 1, 1, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 0.5, 0.5, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 128)
})

test1 = {
    i1: [1, 2, 3, 4],
    o1: [1]
}

Example(test1, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test1, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 2: RESIZE_NEAREST_NEIGHBOR_2, w = 3, h = 3
i1 = Input("in", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{1, 3, 3, 1}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 3, 3, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 1.5, 1.5, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 0)
})

test2 = {
    i1: [1, 2, 3, 4],
    o1: [1, 1, 2, 1, 1, 2, 3, 3, 4]
}

Example(test2, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test2, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 3: RESIZE_NEAREST_NEIGHBOR_3, w = 2, h = 2
i1 = Input("in", "TENSOR_FLOAT32", "{1, 3, 3, 1}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 2, 2, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 0.8, 0.8, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 100)
})

test3 = {
    i1: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    o1: [1, 2, 4, 5]
}

Example(test3, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test3, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 4: RESIZE_NEAREST_NEIGHBOR_4, w = 5, h = 2
i1 = Input("in", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{1, 2, 5, 1}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 5, 2, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 2.6, 1.1, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 100)
})

test4 = {
    i1: [1, 2, 3, 4],
    o1: [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
}

Example(test4, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test4, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 5: RESIZE_NEAREST_NEIGHBOR_5, w = 3, h = 3
i1 = Input("in", "TENSOR_FLOAT32", "{1, 4, 4, 1}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{1, 3, 3, 1}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 3, 3, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 0.9, 0.9, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 100)
})

test5 = {
    i1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    o1: [1, 2, 3, 5, 6, 7, 9, 10, 11]
}

Example(test5, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test5, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 6: RESIZE_NEAREST_NEIGHBOR_6, w = 2, h = 5
i1 = Input("in", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{1, 5, 2, 1}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 2, 5, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 1.4, 2.8, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 100)
})

test6 = {
    i1: [1, 2, 3, 4],
    o1: [1, 2, 1, 2, 1, 2, 3, 4, 3, 4]
}

Example(test6, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test6, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 7: RESIZE_NEAREST_NEIGHBOR_7, w = 4, h = 4
i1 = Input("in", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{1, 4, 4, 1}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 4, 4, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 2.0, 2.0, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 100)
})

test7 = {
    i1: [1, 2, 3, 4],
    o1: [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
}

Example(test7, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test7, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 8: RESIZE_NEAREST_NEIGHBOR_8, w = 3, h = 3
i1 = Input("in", "TENSOR_FLOAT32", "{2, 2, 2, 2}") # input 0
o1 = Output("out", "TENSOR_FLOAT32", "{2, 3, 3, 2}") # output 0
model_shape = Model("shape").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 3, 3, layout).To(o1)
model_scale = Model("scale").Operation("RESIZE_NEAREST_NEIGHBOR", i1, 1.6, 1.8, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 100),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 100)
})

test8 = {
    i1: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    o1: [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2,
         3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6,
         5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8]
}

Example(test8, model=model_shape).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")
Example(test8, model=model_scale).AddNchw(i1, o1, layout).AddVariations("relaxed", quant8, "float16")


# TEST 8: zero-sized input, resize by output shape

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

# RESIZE_NEAREST_NEIGHBOR op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 3, 3, 1}") # out
model = model.Operation("RESIZE_NEAREST_NEIGHBOR", zero_sized, 3, 3, layout).To(o3)

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


# TEST 9: zero-sized input, resize by scale

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

# RESIZE_NEAREST_NEIGHBOR op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 3, 3, 1}") # out
model = model.Operation("RESIZE_NEAREST_NEIGHBOR", zero_sized, 1.6, 1.6, layout).To(o3)

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
