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

i = Input("op1", "TENSOR_FLOAT32", "{2, 2, 2, 5}") # input 0
o = Output("op2", "TENSOR_FLOAT32", "{2, 2, 2, 5}") # output 0
axis = Int32Scalar("axis", -1) # last axis

# Additional data type
quant8 = DataTypeConverter().Identify({
    i: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    o: ("TENSOR_QUANT8_ASYMM", 1./256, 0)
})

example1 = {
    i: [17., 16., 15., 14.,  1.,
        -1., -2., -3., -4., -17.] * 4,
    o: [0.643914213228014,
        0.236882800924671,
        0.087144312427294,
        0.032058600957022,
        7.246299848982885e-08] * 8
}
example2 = {
    i: [1., 2., 3., 4., 5., -1., -2., -3., -4., -5.] * 4,
    o: [0.2] * 40
}

# TEST 1: All dimensions other than 2 or 4, without axis parameter
# beta = 1.0
Model().Operation("SOFTMAX", i, 1.0).To(o)
Example(example1).AddVariations("relaxed", "float16", quant8).AddDims([1, 3], i, o)
# beta = 0.000001
Model().Operation("SOFTMAX", i, 0.000001).To(o)
Example(example2).AddVariations("relaxed", "float16", quant8).AddDims([1, 3], i, o)

# TEST 2: All dimensions, with all possible axis parameter
# beta = 1.0
Model("axis").Operation("SOFTMAX", i, 1.0, axis).To(o)
Example(example1).AddVariations("relaxed", "float16", quant8).AddAllDimsAndAxis(i, o, axis)
# beta = 0.000001
Model("axis").Operation("SOFTMAX", i, 0.000001, axis).To(o)
Example(example2).AddVariations("relaxed", "float16", quant8).AddAllDimsAndAxis(i, o, axis)

# SOFTMAX of rank 4 and TENSOR_FLOAT32 and TENSOR_QUANT8_ASYMM data type is introduced in V1_0.
Example.SetVersion("V1_0", "softmax_v1_2", "softmax_v1_2_quant8", \
                           "softmax_v1_2_2", "softmax_v1_2_quant8_2")


# TEST 3: zero-sized input

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

# SOFTMAX op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 2, 2, 1}") # out
model = model.Operation("SOFTMAX", zero_sized, 1.0).To(o3)

quant8 = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    zero_sized: ("TENSOR_QUANT8_ASYMM", 0.1, 128),
    o3: ("TENSOR_QUANT8_ASYMM", 1./256, 0)
})

# Create test case with dummy values.
Example({
    i1: [1],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddVariations("relaxed", quant8, "float16")
