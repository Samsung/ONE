#
# Copyright (C) 2018 The Android Open Source Project
# Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

'''
Testing Tanh op when the input is dynamic.

      input [2x2]    shape [2]  (value of shape will be 2, 2)
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be 2x2 at execution time
          |
         Tanh
          |
        output [2x2], dynamic tensor
'''

import dynamic_tensor

model = Model()

model_input_shape = [2, 2]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

# write Tanh test.

# note output shape is used as the shape of expected output.
model_output = Output("output", "TENSOR_FLOAT32", "{2, 2}")

model.Operation("TANH", test_node_input).To(model_output)

model_input_data = [-1, 0, 1, 10] # input value list to Tanh
model_output_data = [-.761594156, 0, .761594156, 0.999999996] # output value list of Tanh

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
#.AddVariations("relaxed", "float16")
