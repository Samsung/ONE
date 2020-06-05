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

# refer to tanh_v1_dynamic.mod.py about the structore

# This adds reshape as the first op in a model and
# returns output of reshape, which is dynamic tensor

'''
Testing Pack op when the input is dynamic.
       input3[6, 4]  shape [2]  (value of shape will be [6, 4])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          |        input1[6, 4]    input2[6, 4]
          |             |             |
          +-------------+-------------+
          |
          |
          | dynamic tensor at compilation time but the shape will be [6, 4] at execution time
          |
         Pack
          |
        output (dynamic tensor, [6, 3, 4] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [6, 4]

input1 = Input("input1", "TENSOR_FLOAT32", "{6, 4}")
input2 = Input("input2", "TENSOR_FLOAT32", "{6, 4}")

num = Int32Scalar("num_tensors", 3)

axis = Int32Scalar("axis", 1)

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()
# write PACK_EX test. input is `test_input`

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_FLOAT32", "{6, 3, 4}")

model.Operation("PACK_EX", test_node_input, input1, input2, num, axis).To(model_output)

model_input_data = [0.3, 1.0, 2.0, 3.0,
            4.0, 5.5, 6.3, 7.2,
            8.22, 9.8, 10.3, 11.0,
            12.22, 13.2, 14.44, 15.32,
            16.55, 17.33, 18.1, 19.0,
            20.32, 21.9, 22.1, 23.22]

input1_data = [24.22, 25.1, 26.0, 27.12,
            28.32, 29.11, 30.0, 31.98,
            32.99, 33.11, 34.1, 35.123,
            36.21, 37.22, 38.23, 39.76,
            40.1, 41.43, 42.34, 43.1,
            44.123, 45.43, 46.1, 47.1]

input2_data = [48.0, 49.76, 50.0, 51.1,
                    52.22, 53.12, 54.1, 55.5,
                    56.5, 57.4, 58.1, 59.23,
                    60.2, 61.12, 62.11, 63.34,
                    64.11, 65.1, 66.43, 67.1,
                    68.1, 69.34, 70.11, 71.45]

model_output_data = [0.3, 1.0, 2.0, 3.0,
                    24.22, 25.1, 26.0, 27.12,
                    48.0, 49.76, 50.0, 51.1,
                    4.0, 5.5, 6.3, 7.2,
                    28.32, 29.11, 30.0, 31.98,
                    52.22, 53.12, 54.1, 55.5,
                    8.22, 9.8, 10.3, 11.0,
                    32.99, 33.11, 34.1, 35.123,
                    56.5, 57.4, 58.1, 59.23,
                    12.22, 13.2, 14.44, 15.32,
                    36.21, 37.22, 38.23, 39.76,
                    60.2, 61.12, 62.11, 63.34,
                    16.55, 17.33, 18.1, 19.0,
                    40.1, 41.43, 42.34, 43.1,
                    64.11, 65.1, 66.43, 67.1,
                    20.32, 21.9, 22.1, 23.22,
                    44.123, 45.43, 46.1, 47.1,
                    68.1, 69.34, 70.11, 71.45]

Example({
    dynamic_layer.getModelInput() : model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,
    input1 : input1_data,
    input2 : input2_data,

    model_output: model_output_data,
})
