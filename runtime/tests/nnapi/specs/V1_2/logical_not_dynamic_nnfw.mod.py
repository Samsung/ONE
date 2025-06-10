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
Testing LogicalNot op when the input is dynamic.

      input [1, 2, 3]  shape [3]  (value of shape will be [1, 2, 3])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 2, 3] at execution time
          |
         LogicalNot
          |
        output (dynamic tensor, [1, 2, 3] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [1, 2, 3]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_BOOL8")

test_node_input = dynamic_layer.getTestNodeInput()

# write LOGICAL_NOT test. input is `test_input`

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_BOOL8", "{1, 2, 3}")

model.Operation("LOGICAL_NOT", test_node_input).To(model_output)

model_input_data = [True, False, True, False, True, True]
model_output_data = [False, True, False, True, False, False]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
