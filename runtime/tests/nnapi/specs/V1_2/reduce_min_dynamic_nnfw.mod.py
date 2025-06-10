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

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")
test_node_input = dynamic_layer.getTestNodeInput()


axis = Parameter("axis", "TENSOR_INT32", "{4}", [1, 0, -3, -3])
keepDims = False

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_FLOAT32", "{2}")

model.Operation("REDUCE_MIN", test_node_input, axis, keepDims).To(model_output)

model_input_data = [23.0,  24.0,
                    13.0,  22.0,
                     5.0,  18.0,

                     7.0,  8.0,
                     9.0, 15.0,
                    11.0, 12.0,

                    3.0, 14.0,
                    10.0, 16.0,
                    17.0, 6.0,

                    19.0, 20.0,
                    21.0, 4.0,
                     1.0, 2.0]

model_output_data = [1.0, 2.0]

Example({
    dynamic_layer.getModelInput() : model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
