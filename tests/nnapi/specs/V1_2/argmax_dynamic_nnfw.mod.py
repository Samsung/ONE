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
Testing Argmax op when the input is dynamic.

      input [1, 2, 2, 2]  shape [4]  (value of shape will be [1, 2, 2, 2])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 2, 2, 2] at execution time
          |
         Argmax
          |
        output (dynamic tensor, [1, 2, 2] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [1, 2, 2, 2]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape)

test_node_input = dynamic_layer.getTestNodeInput()

axis = Parameter("axis", "TENSOR_INT32", "{1}", [3])

# write ARGMAX test. input is `test_input`

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_INT32", "{1, 2, 2}")

model.Operation("ARGMAX", test_node_input, axis).To(model_output)

model_input_data = [1.0, 2.0, 7.0, 8.0,
                    1.0, 9.0, 7.0, 3.0]

model_output_data = [1, 1, 1, 0]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
