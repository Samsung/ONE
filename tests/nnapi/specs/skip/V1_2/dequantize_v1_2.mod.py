#
# Copyright (C) 2019 The Android Open Source Project
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


def test(name, input0, output0, input0_data, output0_data):
  model = Model().Operation("DEQUANTIZE", input0).To(output0)
  example = Example({
      input0: input0_data,
      output0: output0_data,
  },
                    model=model,
                    name=name).AddVariations("relaxed", "float16")


test(
    name="1d_quant8_asymm",
    input0=Input("input0", "TENSOR_QUANT8_ASYMM", "{10}, 0.5, 127"),
    output0=Output("output0", "TENSOR_FLOAT32", "{10}"),
    input0_data=[0, 1, 2, 3, 4, 251, 252, 253, 254, 255],
    output0_data=[-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64],
)

test(
    name="2d_quant8_asymm",
    input0=Input("input0", "TENSOR_QUANT8_ASYMM", "{2, 5}, 0.5, 127"),
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 5}"),
    input0_data=[0, 1, 2, 3, 4, 251, 252, 253, 254, 255],
    output0_data=[-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64],
)

test(
    name="3d_quant8_symm",
    input0=Input("input0", "TENSOR_QUANT8_SYMM", "{2, 2, 2}, 0.5, 0"),
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 2, 2}"),
    input0_data=[-128, -127, -126, -125, 124, 125, 126, 127],
    output0_data=[-64, -63.5, -63, -62.5, 62, 62.5, 63, 63.5],
)

test(
    name="4d_quant8_symm",
    input0=Input("input0", "TENSOR_QUANT8_SYMM", "{2, 1, 2, 2}, 0.5, 0"),
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 1, 2, 2}"),
    input0_data=[-128, -127, -126, -125, 124, 125, 126, 127],
    output0_data=[-64, -63.5, -63, -62.5, 62, 62.5, 63, 63.5],
)

test(
    name="3d_per_channel_first_dim",
    input0=Input(
        "input0", ("TENSOR_QUANT8_SYMM_PER_CHANNEL", [2, 3, 4], 0, 0),
        extraParams=SymmPerChannelQuantParams(channelDim=0, scales=[2., 0.5])),
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 3, 4}"),
    input0_data=[
        -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117,
        116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
    ],
    output0_data=[
        -256, -254, -252, -250, -248, -246, -244, -242, -240, -238, -236, -234,
        58., 58.5, 59., 59.5, 60., 60.5, 61., 61.5, 62., 62.5, 63., 63.5
    ],
)

test(
    name="3d_per_channel_second_dim",
    input0=Input(
        "input0", ("TENSOR_QUANT8_SYMM_PER_CHANNEL", [2, 3, 4], 0, 0),
        extraParams=SymmPerChannelQuantParams(
            channelDim=1, scales=[2., 1., 0.5])),
    output0=Output("output0", "TENSOR_FLOAT32", "{2, 3, 4}"),
    input0_data=[
        -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117,
        116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127
    ],
    output0_data=[
        -256., -254., -252., -250., -124., -123., -122., -121., -60., -59.5,
        -59., -58.5, 232., 234., 236., 238., 120., 121., 122., 123., 62., 62.5,
        63., 63.5
    ],
)

# DEQUANTIZE from TENSOR_QUANT8_ASYMM to TENSOR_FLOAT32 is introduced in V1_0.
Example.SetVersion("V1_0", "dequantize_v1_2_1d_quant8_asymm", "dequantize_v1_2_2d_quant8_asymm")

# FLOAT16
model = Model()
i1 = Input("op1",  "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 1.f, 0")
i2 = Output("op2", "TENSOR_FLOAT16", "{1, 2, 2, 1}")
model = model.Operation("DEQUANTIZE", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 32, 128, 255]}

output0 = {i2: # output 0
           [0.0, 32.0, 128.0, 255.0]}

# Instantiate an example
Example((input0, output0))


# Zero-sized input

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_QUANT8_ASYMM", "{1, 2}, 0.1f, 128", [137, 129]) # scores
p2 = Parameter("roi", "TENSOR_QUANT16_ASYMM", "{1, 8}, 0.125f, 0", [8, 8, 80, 80, 0, 0, 80, 80]) # roi
o1 = Output("scoresOut", "TENSOR_QUANT8_ASYMM", "{0}, 0.1f, 128") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_QUANT16_ASYMM", "{0, 4}, 0.125f, 0") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
layout = BoolScalar("layout", False) # NHWC
i1 = Input("in", "TENSOR_QUANT8_ASYMM", "{1, 1, 1, 1}, 0.1f, 128")
zero_sized = Internal("featureMap", "TENSOR_QUANT8_ASYMM", "{0, 2, 2, 1}, 0.1f, 128")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# DEQUANTIZE op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 2, 2, 1}") # out
model = model.Operation("DEQUANTIZE", zero_sized).To(o3)

float16 = DataTypeConverter().Identify({o3: ("TENSOR_FLOAT16",)})

# Create test case with dummy values.
Example({
    i1: [1],
    o1: [0],
    o2: [0],
    o3: [0],
}).AddVariations("relaxed", float16)
