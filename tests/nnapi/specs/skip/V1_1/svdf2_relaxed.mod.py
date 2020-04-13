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

batches = 2
features = 8
rank = 2
units = int(features / rank)
input_size = 3
memory_size = 10

model = Model()

input = Input("input", "TENSOR_FLOAT32", "{%d, %d}" % (batches, input_size))
weights_feature = Input("weights_feature", "TENSOR_FLOAT32", "{%d, %d}" % (features, input_size))
weights_time = Input("weights_time", "TENSOR_FLOAT32", "{%d, %d}" % (features, memory_size))
bias = Input("bias", "TENSOR_FLOAT32", "{%d}" % (units))
state_in = Input("state_in", "TENSOR_FLOAT32", "{%d, %d}" % (batches, memory_size*features))
rank_param = Int32Scalar("rank_param", rank)
activation_param = Int32Scalar("activation_param", 0)
state_out = IgnoredOutput("state_out", "TENSOR_FLOAT32", "{%d, %d}" % (batches, memory_size*features))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (batches, units))

model = model.Operation("SVDF", input, weights_feature, weights_time, bias, state_in,
                        rank_param, activation_param).To([state_out, output])
model = model.RelaxedExecution(True)

input0 = {
    input: [],
    weights_feature: [
      -0.31930989, 0.0079667,   0.39296314,  0.37613347,
      0.12416199,  0.15785322,  0.27901134,  0.3905206,
      0.21931258,  -0.36137494, -0.10640851, 0.31053296,
      -0.36118156, -0.0976817,  -0.36916667, 0.22197971,
      0.15294972,  0.38031587,  0.27557442,  0.39635518,
      -0.21580373, -0.06634006, -0.02702999, 0.27072677
    ],
    weights_time: [
      -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657,

       -0.14884081, 0.19931212,  -0.36002168, 0.34663299,  -0.11405486,
       0.12672701,  0.39463779,  -0.07886535, -0.06384811, 0.08249187,

       -0.26816407, -0.19905911, 0.29211238,  0.31264046,  -0.28664589,
       0.05698794,  0.11613581,  0.14078894,  0.02187902,  -0.21781836,

       -0.15567942, 0.08693647,  -0.38256618, 0.36580828,  -0.22922277,
       -0.0226903,  0.12878349,  -0.28122205, -0.10850525, -0.11955214,

       0.27179423,  -0.04710215, 0.31069002,  0.22672787,  0.09580326,
       0.08682203,  0.1258215,   0.1851041,   0.29228821,  0.12366763
    ],
    bias: [],
    state_in: [0 for _ in range(batches * memory_size * features)],
}

test_inputs = [
    0.12609188,  -0.46347019, -0.89598465,
    0.35867718,  0.36897406,  0.73463392,

    0.14278367,  -1.64410412, -0.75222826,
    -0.57290924, 0.12729003,  0.7567004,

    0.49837467,  0.19278903,  0.26584083,
    0.17660543,  0.52949083,  -0.77931279,

    -0.11186574, 0.13164264,  -0.05349274,
    -0.72674477, -0.5683046,  0.55900657,

    -0.68892461, 0.37783599,  0.18263303,
    -0.63690937, 0.44483393,  -0.71817774,

    -0.81299269, -0.86831826, 1.43940818,
    -0.95760226, 1.82078898,  0.71135032,

    -1.45006323, -0.82251364, -1.69082689,
    -1.65087092, -1.89238167, 1.54172635,

    0.03966608,  -0.24936394, -0.77526885,
    2.06740379,  -1.51439476, 1.43768692,

    0.11771342,  -0.23761693, -0.65898693,
    0.31088525,  -1.55601168, -0.87661445,

    -0.89477462, 1.67204106,  -0.53235275,
    -0.6230064,  0.29819036,  1.06939757,
]

golden_outputs = [
    -0.09623547, -0.10193135, 0.11083051,  -0.0347917,
    0.1141196,   0.12965347,  -0.12652366, 0.01007236,

    -0.16396809, -0.21247184, 0.11259045,  -0.04156673,
    0.10132131,  -0.06143532, -0.00924693, 0.10084561,

    0.01257364,  0.0506071,   -0.19287863, -0.07162561,
    -0.02033747, 0.22673416,  0.15487903,  0.02525555,

    -0.1411963,  -0.37054959, 0.01774767,  0.05867489,
    0.09607603,  -0.0141301,  -0.08995658, 0.12867066,

    -0.27142537, -0.16955489, 0.18521598,  -0.12528358,
    0.00331409,  0.11167502,  0.02218599,  -0.07309391,

    0.09593632,  -0.28361851, -0.0773851,  0.17199151,
    -0.00075242, 0.33691186,  -0.1536046,  0.16572715,

    -0.27916506, -0.27626723, 0.42615682,  0.3225764,
    -0.37472126, -0.55655634, -0.05013514, 0.289112,

    -0.24418658, 0.07540751,  -0.1940318,  -0.08911639,
    0.00732617,  0.46737891,  0.26449674,  0.24888524,

    -0.17225097, -0.54660404, -0.38795233, 0.08389944,
    0.07736043,  -0.28260678, 0.15666828,  1.14949894,

    -0.57454878, -0.64704704, 0.73235172,  -0.34616736,
    0.21120001,  -0.22927976, 0.02455296,  -0.35906726,
]

output0 = {state_out: [0 for _ in range(batches * memory_size * features)],
           output: []}

# TODO: enable more data points after fixing the reference issue
for i in range(1):
  batch_start = i * input_size * batches
  batch_end = batch_start + input_size * batches
  input0[input] = test_inputs[batch_start:batch_end]
  golden_start = i * units * batches
  golden_end = golden_start + units * batches
  output0[output] = golden_outputs[golden_start:golden_end]
  Example((input0, output0))
