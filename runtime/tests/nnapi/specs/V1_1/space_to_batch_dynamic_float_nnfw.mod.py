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
Testing SpaceToBatchND op when the input is dynamic.
      input [1, 4 ,2, 1]  shape [4]  (value of shape will be [1, 4 ,2, 1])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 4 ,2, 1] at execution time
          |
         SpaceToBatchND
          |
        output (dynamic tensor, [2, 3] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [1, 4, 2, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(
    model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

i2 = Parameter("block_size", "TENSOR_INT32", "{2}", [3, 2])
i3 = Parameter("paddings", "TENSOR_INT32", "{2, 2}", [1, 1, 2, 4])

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_FLOAT32", "{6, 2, 4, 1}")

model.Operation("SPACE_TO_BATCH_ND", test_node_input, i2, i3).To(model_output)


model_input_data = [1, 2, 3, 4, 5, 6, 7, 8]

model_output_data = [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
                     0, 1, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0,
                     0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput(): model_input_shape,

    model_output: model_output_data,
})
