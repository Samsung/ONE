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
units = 4
input_size = 3
memory_size = 10

model = Model()

input = Input("input", "TENSOR_FLOAT32", "{%d, %d}" % (batches, input_size))
weights_feature = Input("weights_feature", "TENSOR_FLOAT32", "{%d, %d}" % (units, input_size))
weights_time = Input("weights_time", "TENSOR_FLOAT32", "{%d, %d}" % (units, memory_size))
bias = Input("bias", "TENSOR_FLOAT32", "{%d}" % (units))
state_in = Input("state_in", "TENSOR_FLOAT32", "{%d, %d}" % (batches, memory_size*units))
rank_param = Int32Scalar("rank_param", 1)
activation_param = Int32Scalar("activation_param", 0)
state_out = Output("state_out", "TENSOR_FLOAT32", "{%d, %d}" % (batches, memory_size*units))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (batches, units))

model = model.Operation("SVDF", input, weights_feature, weights_time, bias, state_in,
                        rank_param, activation_param).To([state_out, output])
model = model.RelaxedExecution(True)

input0 = {
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
    bias: [],
}

input0[input] = [
  0.14278367,  -1.64410412, -0.75222826,
  0.14278367,  -1.64410412, -0.75222826,
]
input0[state_in]  = [
  0, 0, 0, 0,
  0, 0, 0, 0,
  0.119996, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, -0.166701, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  -0.44244, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0.0805206, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0.119996, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, -0.166701, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  -0.44244, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0.0805206, 0,
]
output0 = {
    state_out : [
  0, 0, 0, 0,
  0, 0, 0, 0.119996,
  0.542235, 0, 0, 0,
  0, 0, 0, 0,
  0, -0.166701, -0.40465, 0,
  0, 0, 0, 0,
  0, 0, 0, -0.44244,
  -0.706995, 0, 0, 0,
  0, 0, 0, 0,
  0, 0.0805206, 0.137515, 0,
  0, 0, 0, 0,
  0, 0, 0, 0.119996,
  0.542235, 0, 0, 0,
  0, 0, 0, 0,
  0, -0.166701, -0.40465, 0,
  0, 0, 0, 0,
  0, 0, 0, -0.44244,
  -0.706995, 0, 0, 0,
  0, 0, 0, 0,
  0, 0.0805206, 0.137515, 0,
    ],
    output : [
  0.068281,    -0.162217,  -0.152268, 0.00323521,
  0.068281,    -0.162217,  -0.152268, 0.00323521,
    ]
}

Example((input0, output0))
