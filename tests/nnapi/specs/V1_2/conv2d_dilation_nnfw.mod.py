#
# Copyright (C) 2018 The Android Open Source Project
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

# TEST 1: dilation set to 1 (default)
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [.25, .25, .25, .25])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
Model().Operation("CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 0, layout, 1, 1).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    f1: ("TENSOR_QUANT8_ASYMM", 0.125, 0),
    b1: ("TENSOR_INT32", 0.0625, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.125, 0)
})

# Instantiate an example
example = Example({
    i1: [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
    o1: [.875, .875, .875, .875]
}).AddInput(f1, b1).AddVariations("relaxed", quant8, "float16")


# TEST 2: dilation set to 3
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 9, 9, 1}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
Model().Operation("CONV_2D", i2, f2, b2, 0, 0, 0, 0, 1, 1, 0, layout, 3, 3).To(o2)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    f2: ("TENSOR_QUANT8_ASYMM", 0.125, 0),
    b2: ("TENSOR_INT32", 0.0625, 0),
    o2: ("TENSOR_QUANT8_ASYMM", 0.125, 0)
})

# Instantiate an example
example = Example({
    i2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    o2: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
}).AddInput(f2, b2).AddVariations("relaxed", quant8, "float16")
