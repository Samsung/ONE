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

model_input_shape = [6]

axis = Int32Scalar("axis", 0)
num_splits = Int32Scalar("num_splits", 3)

output0 = Output("output0", "TENSOR_FLOAT32", "{2}")
output1 = Output("output1", "TENSOR_FLOAT32", "{2}")
output2 = Output("output2", "TENSOR_FLOAT32", "{2}")

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

model.Operation("SPLIT", test_node_input, axis, num_splits).To([output0, output1, output2])

model_input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

output_0_data = [1.0, 2.0]
output_1_data = [3.0, 4.0]
output_2_data = [5.0, 6.0]

Example(
  {
    dynamic_layer.getModelInput() : model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    output0: output_0_data,
    output1: output_1_data,
    output2: output_2_data,
  })
