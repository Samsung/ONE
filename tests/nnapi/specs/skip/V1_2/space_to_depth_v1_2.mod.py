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

# TEST 1: SPACE_TO_DEPTH_NCHW_1, block_size = 2
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 1, 8}")
Model().Operation("SPACE_TO_DEPTH", i1, 2, layout).To(o1)

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


# TEST 2: SPACE_TO_DEPTH_NCHW_2, block_size = 2
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 1}")
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
Model().Operation("SPACE_TO_DEPTH", i2, 2, layout).To(o2)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    o2: ("TENSOR_QUANT8_ASYMM", 0.5, 128)
})

# Instantiate an example
example = Example({
    i2: [1., 2., 5., 6., 3., 4., 7., 8., 9., 10., 13., 14., 11., 12., 15., 16.],
    o2: [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.]
}).AddNchw(i2, o2, layout).AddVariations("relaxed", "float16", quant8)


# TEST 3: SPACE_TO_DEPTH_NCHW_3, block_size = 2
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 2}")
o3 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 8}")
Model().Operation("SPACE_TO_DEPTH", i3, 2, layout).To(o3)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 1.0, 0),
    o3: ("TENSOR_QUANT8_ASYMM", 1.0, 0)
})

# Instantiate an example
example = Example({
    i3: [10,   20,  11,  21,  12,  22, 13,   23,
         14,   24,  15,  25,  16,  26, 17,   27,
         18,   28,  19,  29, 110, 210, 111, 211,
        112,  212, 113, 213, 114, 214, 115, 215],
    o3: [10,   20,  11,  21, 14,   24,  15,  25,
         12,   22,  13,  23, 16,   26,  17,  27,
         18,   28,  19,  29, 112, 212, 113, 213,
         110, 210, 111, 211, 114, 214, 115, 215]
}).AddNchw(i3, o3, layout).AddVariations("relaxed", "float16", quant8)
