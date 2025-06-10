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

# TEST 1: SPACE_TO_BATCH_NCHW_1, block_size = [2, 2]
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
pad1 = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [0, 0, 0, 0])
o1 = Output("op4", "TENSOR_FLOAT32", "{4, 1, 1, 2}")
Model().Operation("SPACE_TO_BATCH_ND", i1, [2, 2], pad1, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 0)
})

# Instantiate an example
example = Example({
    i1: [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1],
    o1: [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1]
}).AddNchw(i1, o1, layout).AddVariations("relaxed", "float16", quant8)


# TEST 2: SPACE_TO_BATCH_NCHW_2, block_size = [2, 2]
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 1}")
o2 = Output("op4", "TENSOR_FLOAT32", "{4, 2, 2, 1}")
Model().Operation("SPACE_TO_BATCH_ND", i2, [2, 2], pad1, layout).To(o2)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    o2: ("TENSOR_QUANT8_ASYMM", 0.5, 0)
})

# Instantiate an example
example = Example({
    i2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    o2: [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16]
}).AddNchw(i2, o2, layout).AddVariations("relaxed", "float16", quant8)


# TEST 3: SPACE_TO_BATCH_NCHW_3, block_size = [3, 2]
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 5, 2, 1}")
pad3 = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [1, 0, 2, 0])
o3 = Output("op4", "TENSOR_FLOAT32", "{6, 2, 2, 1}")
Model().Operation("SPACE_TO_BATCH_ND", i3, [3, 2], pad3, layout).To(o3)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    o3: ("TENSOR_QUANT8_ASYMM", 0.5, 128)
})

# Instantiate an example
example = Example({
    i3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    o3: [0, 0, 0, 5, 0, 0, 0, 6, 0, 1, 0, 7,
         0, 2, 0, 8, 0, 3, 0, 9, 0, 4, 0, 10]
}).AddNchw(i3, o3, layout).AddVariations("relaxed", "float16", quant8)


# TEST 4: SPACE_TO_BATCH_NCHW_4, block_size = [3, 2]
i4 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 2, 1}")
pad4 = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [1, 1, 2, 4])
o4 = Output("op4", "TENSOR_FLOAT32", "{6, 2, 4, 1}")
Model().Operation("SPACE_TO_BATCH_ND", i4, [3, 2], pad4, layout).To(o4)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    o4: ("TENSOR_QUANT8_ASYMM", 0.25, 128)
})

# Instantiate an example
example = Example({
    i4: [1, 2, 3, 4, 5, 6, 7, 8],
    o4: [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
         0, 1, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0,
         0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0]
}).AddNchw(i4, o4, layout).AddVariations("relaxed", "float16", quant8)
