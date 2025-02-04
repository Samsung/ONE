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

batches = 2
features = 4
rank = 1
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
        -0.31930989, -0.36118156, 0.0079667, 0.37613347,
      0.22197971, 0.12416199, 0.27901134, 0.27557442,
      0.3905206, -0.36137494, -0.06634006, -0.10640851
    ],
    weights_time: [
        -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
      0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
      -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
      0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
      -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657
    ],
    bias: [1.0, 2.0, 3.0, 4.0],
    state_in: [0 for _ in range(batches * memory_size * features)],
}

test_inputs = [
    0.12609188,  -0.46347019, -0.89598465,
    0.12609188,  -0.46347019, -0.89598465,

    0.14278367,  -1.64410412, -0.75222826,
    0.14278367,  -1.64410412, -0.75222826,

    0.49837467,  0.19278903,  0.26584083,
    0.49837467,  0.19278903,  0.26584083,

    -0.11186574, 0.13164264,  -0.05349274,
    -0.11186574, 0.13164264,  -0.05349274,

    -0.68892461, 0.37783599,  0.18263303,
    -0.68892461, 0.37783599,  0.18263303,

    -0.81299269, -0.86831826, 1.43940818,
    -0.81299269, -0.86831826, 1.43940818,

    -1.45006323, -0.82251364, -1.69082689,
    -1.45006323, -0.82251364, -1.69082689,

    0.03966608,  -0.24936394, -0.77526885,
    0.03966608,  -0.24936394, -0.77526885,

    0.11771342,  -0.23761693, -0.65898693,
    0.11771342,  -0.23761693, -0.65898693,

    -0.89477462, 1.67204106,  -0.53235275,
    -0.89477462, 1.67204106,  -0.53235275
]

golden_outputs = [
    1.014899,    1.9482339,  2.856275,  3.99728117,
    1.014899,    1.9482339,  2.856275,  3.99728117,

    1.068281,    1.837783,   2.847732,  4.00323521,
    1.068281,    1.837783,   2.847732,  4.00323521,

    0.9682179,   1.9666911,  3.0609602, 4.0333759,
    0.9682179,   1.9666911,  3.0609602, 4.0333759,

    0.99376901,  1.922299,   2.608807,  3.9863309,
    0.99376901,  1.922299,   2.608807,  3.9863309,

    1.201551,    1.835393,   2.820538,  3.9407261,
    1.201551,    1.835393,   2.820538,  3.9407261,

    1.0886511,   1.9124599,  2.730717,  4.0281379,
    1.0886511,   1.9124599,  2.730717,  4.0281379,

    0.798826,    1.413855,   2.371376,  3.9669588,
    0.798826,    1.413855,   2.371376,  3.9669588,

    0.9160904,   1.700671,   3.108746,  4.109808,
    0.9160904,   1.700671,   3.108746,  4.109808,

    1.419114,    1.762176,   2.577373,  4.175115,
    1.419114,    1.762176,   2.577373,  4.175115,

    1.36726,     1.477697,   2.543498,  3.824525,
    1.36726,     1.477697,   2.543498,  3.824525
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
