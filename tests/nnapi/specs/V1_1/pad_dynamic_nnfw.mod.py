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
Testing Pad op when the input is dynamic.
      input [4]  shape [1, 2, 2, 1]  (value of shape will be [1, 2, 2, 1])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 2, 2, 1] at execution time
          |
         Pad
          |
        output (dynamic tensor, [3, 4, 5, 3] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [1, 2, 2, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

pad = Parameter("pad", "TENSOR_INT32", "{4, 2}", [1, 1, 1, 1, 2, 1, 1, 1])

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_FLOAT32", "{3, 4, 5, 3}")

model.Operation("PAD", test_node_input, pad).To(model_output)

model_input_data = [1.0, 2.0,
                    3.0, 4.0]
model_output_data = [0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 2.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 3.0, 0.0,
                     0.0, 4.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})

