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

# TEST 1: DEPTHWISE_CONV2D_NCHW, pad = 0, stride = 1, cm = 2, act = none
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 2}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0., .2, 0., .25, 0., 0., .3, .25, 0., 0., 0., .25, .1, 0., 0.])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 2, 0, layout).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    f1: ("TENSOR_QUANT8_ASYMM", 0.01, 0),
    b1: ("TENSOR_INT32", 0.005, 0),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 0)
})
channelQuant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    f1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.01, 0.005, 0.01, 0.005])),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.005, 0.0025, 0.005, 0.0025], hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM", 0.1, 0)
})
channelQuant8_mult_gt_1 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.5, 0),
    f1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.01, 0.005, 0.01, 0.005])),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.005, 0.0025, 0.005, 0.0025], hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM", 0.0001, 0)
})

# Instantiate an example
example = Example({
    i1: [10, 21, 10, 22, 10, 23,
         10, 24, 10, 25, 10, 26,
         10, 27, 10, 28, 10, 29],
    o1: [11, 3, 7.2, 10.6,
         11, 3, 7.4, 10.9,
         11, 3, 7.8, 11.5,
         11, 3, 8.0, 11.8]
}).AddNchw(i1, o1, layout).AddInput(f1, b1).AddVariations("relaxed", "float16", channelQuant8, channelQuant8_mult_gt_1, quant8)


# TEST 2: DEPTHWISE_CONV2D_NCHW_2, pad = valid, stride = 1, cm = 2, act = none
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 2, 2}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 1, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i2, f2, b2, 2, 1, 1, 2, 0, layout).To(o2)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    f2: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    b2: ("TENSOR_INT32", 0.25, 0),
    o2: ("TENSOR_QUANT8_ASYMM", 1.0, 100)
})
channelQuant8 = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    f2: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.5, 0.25, 0.5, 0.25])),
    b2: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25, 0.125, 0.25, 0.125], hide=True)),
    o2: ("TENSOR_QUANT8_ASYMM", 1.0, 100)
})

# Instantiate an example
example = Example({
    i2: [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12],
    o2: [71, -34, 99, -20, 91, -26, 127, -4]
}).AddNchw(i2, o2, layout).AddInput(f2, b2).AddVariations("relaxed", "float16", quant8, channelQuant8)


# TEST 3: DEPTHWISE_CONV2D_NCHW_LARGE, pad = 0, stride = 1, cm = 1, act = none
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
f3 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 2}", [.25, 0, .25, 1, .25, 0, .25, 1])
b3 = Parameter("op3", "TENSOR_FLOAT32", "{2}", [100, 200])
o3 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 1, 2}")
Model("large").Operation("DEPTHWISE_CONV_2D", i3, f3, b3, 0, 0, 0, 0, 1, 1, 1, 0, layout).To(o3)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 0.5, 100),
    f3: ("TENSOR_QUANT8_ASYMM", 0.125, 128),
    b3: ("TENSOR_INT32", 0.0625, 0),
    o3: ("TENSOR_QUANT8_ASYMM", 2.0, 128)
})
channelQuant8 = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    f3: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.125, 0.25])),
    b3: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.0625, 0.125], hide=True)),
    o3: ("TENSOR_QUANT8_ASYMM", 2.0, 128)
})

# Instantiate an example
example = Example({
    i3: [10, 21, 10, 22, 10, 23, 10, 24],
    o3: [110, 246]
}).AddNchw(i3, o3, layout).AddInput(f3, b3).AddVariations("relaxed", "float16", quant8, channelQuant8)


# TEST 4: DEPTHWISE_CONV2D_NCHW_LARGE, pad = 0, stride = 1, cm = 1, act = none
i4 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
f4 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0, 10, 50, .25, 1, 20, 50, .25, 0, 30, 50, .25, 1, 40, 50])
b4 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [6000, 7000, 8000, 9000])
o4 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 1, 4}")
Model("large").Operation("DEPTHWISE_CONV_2D", i4, f4, b4, 0, 0, 0, 0, 1, 1, 1, 0, layout).To(o4)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    f4: ("TENSOR_QUANT8_ASYMM", 0.25, 0),
    b4: ("TENSOR_INT32", 0.125, 0),
    o4: ("TENSOR_QUANT8_ASYMM", 50.0, 0)
})
channelQuant8 = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM", 0.5, 128),
    f4: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[1.0, 2.0, 1.0, 1.0])),
    b4: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 1.0, 0.5, 0.5], hide=True)),
    o4: ("TENSOR_QUANT8_ASYMM", 50.0, 0)
})

# Instantiate an example
example = Example({
    i4: [10, 21, 10, 0,
         10, 22, 20, 0,
         10, 23, 30, 0,
         10, 24, 40, 0],
    o4: [6010, 7046, 11000, 9000]
}).AddNchw(i4, o4, layout).AddInput(f4, b4).AddVariations("relaxed", "float16", quant8, channelQuant8)

# TEST 9: quantized with scale product greater than output scale
input_scale = 256.5 / 255
input_zero_point = 127
filter_scale = 256.5 / 255
filter_zero_point = 128
i9 = Input("op1",
           ("TENSOR_QUANT8_ASYMM", [1, 3, 2, 2], input_scale, input_zero_point))
f9 = Parameter(
    "op2",
    ("TENSOR_QUANT8_ASYMM", [1, 2, 2, 4], filter_scale, filter_zero_point), [
        129, 130, 131, 132, 119, 138, 117, 140, 133, 134, 135, 136, 141, 114,
        143, 112
    ])
b9 = Parameter("op3", ("TENSOR_INT32", [4], input_scale * filter_scale, 0),
               [2, 4, 6, 8])
o9 = Output("op4", ("TENSOR_QUANT8_ASYMM", [1, 2, 1, 4], 1.0, 127))
model9 = Model("quant_output_multiplier_gt_1").Operation("DEPTHWISE_CONV_2D", i9, f9, b9, 2, 1, 1, 2,
                           0).To(o9)

# Instantiate an example
example = Example({
    i9: [129, 131, 141, 143, 133, 135, 145, 147, 137, 139, 149, 151],
    o9: [255, 58, 255, 87, 255, 74, 255, 119]
}, model=model9).AddInput(f9, b9).AddVariations("relaxed")
