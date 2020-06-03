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

# TEST 1: AXIS_ALIGNED_BBOX_TRANSFORM
r1 = Input("roi", "TENSOR_FLOAT32", "{5, 4}")
d1 = Input("bboxDeltas", "TENSOR_FLOAT32", "{5, 8}")
b1 = Input("batchSplit", "TENSOR_INT32", "{5}")
i1 = Input("imageInfo", "TENSOR_FLOAT32", "{4, 2}")
o1 = Output("out", "TENSOR_FLOAT32", "{5, 8}")
model1 = Model().Operation("AXIS_ALIGNED_BBOX_TRANSFORM", r1, d1, b1, i1).To(o1)

quant8 = DataTypeConverter().Identify({
    r1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    d1: ("TENSOR_QUANT8_ASYMM", 0.05, 128),
    i1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT16_ASYMM", 0.125, 0)
})

inputs = {
    r1: [100, 150, 400, 430,
         120, 60, 122, 61,
         10, 20, 20, 50,
         50, 120, 150, 250,
         400, 100, 1000, 2000],
    d1: [0.2, 0.2, 0.1, 0.1,
         0.3, -0.1, -0.2, 0.1,
         -0.5, 0.2, 0.2, -0.5,
         -0.1, -0.1, 2.5, 3,
         -0.5, -0.5, 1, 1,
         0.5, 0.5, -1.5, -1.2,
         0.2, 0.2, -3, -4,
         1, -0.5, 0.3, 0.5,
         0.3, -0.2, 1.1, -0.8,
         0.1, 0.05, -0.5, -0.5],
    b1: [0, 1, 2, 2, 3],
    i1: [512, 512,
         128, 256,
         256, 256,
         1024, 512]
}

Example((inputs, {
    o1: [144.224350, 191.276062, 475.775635, 500.723938,
         217.190384, 107.276062, 462.809631, 416.723938,
         118.778594,  60.396736, 121.221406,  61.003266,
         108.617508,  50.357232, 132.982498,  70.442772,
           0.000000,   0.000000,  23.59140714,  60.77422571,
          18.88435 ,  45.48208571,  21.11565   ,  54.51791429,
         117.51063714, 209.80948286, 122.48935143, 212.19050857,
         132.50705143,  12.83312286, 255.99999571, 227.16685714,
            0.       ,  243.1374815,  512.       , 1024.       ,
        512.       ,  568.7958375,  512.       , 1024.       ]
}), model=model1).AddVariations("relaxed", "float16", quant8)


# TEST 2: AXIS_ALIGNED_BBOX_TRANSFORM_ZERO_BATCH
r2 = Input("roi", "TENSOR_FLOAT32", "{5, 4}")
d2 = Input("bboxDeltas", "TENSOR_FLOAT32", "{5, 8}")
b2 = Input("batchSplit", "TENSOR_INT32", "{5}")
i2 = Input("imageInfo", "TENSOR_FLOAT32", "{7, 2}")
o2 = Output("out", "TENSOR_FLOAT32", "{5, 8}")
model2 = Model().Operation("AXIS_ALIGNED_BBOX_TRANSFORM", r2, d2, b2, i2).To(o2)

quant8 = DataTypeConverter().Identify({
    r2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    d2: ("TENSOR_QUANT8_ASYMM", 0.05, 128),
    i2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o2: ("TENSOR_QUANT16_ASYMM", 0.125, 0)
})

inputs = {
    r2: [100, 150, 400, 430,
         120, 60, 122, 61,
         10, 20, 20, 50,
         50, 120, 150, 250,
         400, 100, 1000, 2000],
    d2: [0.2, 0.2, 0.1, 0.1,
         0.3, -0.1, -0.2, 0.1,
         -0.5, 0.2, 0.2, -0.5,
         -0.1, -0.1, 2.5, 3,
         -0.5, -0.5, 1, 1,
         0.5, 0.5, -1.5, -1.2,
         0.2, 0.2, -3, -4,
         1, -0.5, 0.3, 0.5,
         0.3, -0.2, 1.1, -0.8,
         0.1, 0.05, -0.5, -0.5],
    b2: [0, 2, 5, 5, 6],
    i2: [512, 512,
         32, 32,
         128, 256,
         32, 32,
         32, 32,
         256, 256,
         1024, 512]
}

Example((inputs, {
    o2: [144.224350, 191.276062, 475.775635, 500.723938,
         217.190384, 107.276062, 462.809631, 416.723938,
         118.778594,  60.396736, 121.221406,  61.003266,
         108.617508,  50.357232, 132.982498,  70.442772,
           0.000000,   0.000000,  23.59140714,  60.77422571,
          18.88435 ,  45.48208571,  21.11565   ,  54.51791429,
         117.51063714, 209.80948286, 122.48935143, 212.19050857,
         132.50705143,  12.83312286, 255.99999571, 227.16685714,
            0.       ,  243.1374815,  512.       , 1024.       ,
        512.       ,  568.7958375,  512.       , 1024.       ]
}), model=model2).AddVariations("relaxed", "float16", quant8)
