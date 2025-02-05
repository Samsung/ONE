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
Testing Fill op when the input is dynamic.
      input [3]  shape [1]  (value of shape will be [3])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 2, 3] at execution time
          |
        Fill
          |
        output (dynamic tensor, [1, 2, 3] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [3]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_INT32")

test_node_input = dynamic_layer.getTestNodeInput()

model_value = Input("value", "TENSOR_FLOAT32", "{1}")
model_output = Output("output", "TENSOR_FLOAT32", "{1, 2, 3}")

model.Operation("FILL_EX", test_node_input, model_value).To(model_output)

model_input_data = [1, 2, 3]
model_value_data = [1.23]
model_output_data = [1.23, 1.23, 1.23, 1.23, 1.23, 1.23]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput(): model_input_shape,

    model_value: model_value_data,
    model_output: model_output_data,
})
