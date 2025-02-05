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


import dynamic_tensor

model = Model()

model_input_shape = [4, 3, 2]

axis = Input("input1", "TENSOR_INT32", "{1}")

output0 = Output("output0", "TENSOR_FLOAT32", "{4, 3, 2}")


dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

model.Operation("REVERSE_EX", test_node_input, axis).To([output0])

model_input_data = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

output_0_data = [5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                 17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20]

Example(
  {
    dynamic_layer.getModelInput() : model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    axis: [1],

    output0: output_0_data,
  })
