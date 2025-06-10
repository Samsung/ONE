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

# TEST 1: MUL float16
model = Model()
i1 = Input("op1", "TENSOR_FLOAT16", "{3}") # a vector of 3 float16s
i2 = Input("op2", "TENSOR_FLOAT16", "{3}") # another vector of 3 float16s
act = Int32Scalar("act", 0) # an int32_t scalar activation
i3 = Output("op3", "TENSOR_FLOAT16", "{3}")
model = model.Operation("MUL", i1, i2, act).To(i3)
model = model.RelaxedExecution(False)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0009765625, 1.0, 2.5],
          i2: # input 1
          [2, 0.0001, 3.5]}

output0 = {i3: # output 0
           [2.001953125, 0.0001000165, 8.75]}

# Instantiate an example
Example((input0, output0))


# TEST 2: MUL broadcast float16
model = Model()
i1 = Input("op1", "TENSOR_FLOAT16", "{1, 2}")
i2 = Input("op2", "TENSOR_FLOAT16", "{2, 2}")
act = Int32Scalar("act", 0)
i3 = Output("op3", "TENSOR_FLOAT16", "{2, 2}")
model = model.Operation("MUL", i1, i2, act).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2],
          i2: # input 1
          [1, 2, 3, 4]}

output0 = {i3: # output 0
           [1, 4, 3, 8]}

# Instantiate an example
Example((input0, output0))


# TEST 3: MUL, zero-sized input

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

# MUL op with numBatches = 0.
i2 = Parameter("op", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [1, 2, 3, 4]) # weights
o3 = Output("out", "TENSOR_FLOAT32", "{0, 2, 2, 2}") # out
model = model.Operation("MUL", zero_sized, i2, 0).To(o3)

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
