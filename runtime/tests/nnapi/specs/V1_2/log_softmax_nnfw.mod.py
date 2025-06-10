#
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

# Derived from tensorflow/lite/kernels/activations_test.cc

def test(input0, output0, input_data, beta, axis, output_data):
  model = Model().Operation("LOG_SOFTMAX", input0, beta, axis).To(output0)
  Example({
      input0: input_data,
      output0: output_data,
  }, model=model).AddVariations("relaxed", "float16")

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 1, 2, 4}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 1, 2, 4}"),
    input_data=[0, -6, 2, 4,
                3, -2, 10, 1],
    beta=1.0,
    axis=3,
    output_data=[-4.14297, -10.14297, -2.14297, -.142971,
                 -7.00104, -12.00104, -.00104087, -9.00104],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 1, 4, 2}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 1, 4, 2}"),
    input_data=[0, -6,
                2, 4,
                3, -2,
                10, 1],
    beta=1.0,
    axis=-1,
    output_data=[-.00247565, -6.00247,
                 -2.12692, -.126928,
                 -.00671534, -5.00671,
                 -.000123374, -9.00012],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 2, 4, 1}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 2, 4, 1}"),
    input_data=[0, 2, 3, 10,
                -6, 4, -2, 1],
    beta=1.0,
    axis=-3,
    output_data=[-.00247565, -2.12692, -.00671534, -.000123374,
                 -6.00247, -.126928, -5.00671, -9.00012],
)

test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 1, 2, 4}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 1, 2, 4}"),
    input_data=[0, -.6, .2, .4,
                .3, -.2, 1, .1],
    beta=10.0,
    axis=3,
    output_data=[-4.14297, -10.14297, -2.14297, -.142971,
                 -7.00104, -12.00104, -.00104087, -9.00104],
)

def quant8_test(input0, output0, input_data, beta, axis, output_data):
  model = Model().Operation("LOG_SOFTMAX", input0, beta, axis).To(output0)
  quant8 = DataTypeConverter().Identify({
    input0: ["TENSOR_QUANT8_ASYMM", 10 / 255.0],
    output0: ["TENSOR_QUANT8_ASYMM", 16 / 256.0, 255],
  })

  Example({
      input0: input_data,
      output0: output_data,
  }, model=model).AddVariations(quant8)

quant8_test(
    input0=Input("input0", "TENSOR_FLOAT32", "{1, 1, 2, 4}"),
    output0=Output("output0", "TENSOR_FLOAT32", "{1, 1, 2, 4}"),
    input_data=[0, 6, 2, 4,
                3, 2, 10, 1],
    beta=1.0,
    axis=3,
    output_data=[-6.145078, -.145078, -4.145078, -2.145078,
                 -7.001370, -8.001370, -.001370, -9.001370],
)
