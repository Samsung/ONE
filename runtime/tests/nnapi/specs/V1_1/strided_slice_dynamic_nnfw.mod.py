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
Testing StridedSlice op when the input is dynamic.
      input [1, 2, 3, 4]  shape [4]  (value of shape will be [1, 2, 3, 4])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 2, 3, 4] at execution time
          |
         StridedSlice
          |
        output (dynamic tensor, [2, 3] at execution time)
'''

import dynamic_tensor

model = Model()

model_input_shape = [4]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_FLOAT32")

test_node_input = dynamic_layer.getTestNodeInput()

begins = Parameter("begins", "TENSOR_INT32", "{1}", [1])
ends = Parameter("ends", "TENSOR_INT32", "{1}", [3])
strides = Parameter("strides", "TENSOR_INT32", "{1}", [1])
beginMask = Int32Scalar("beginMask", 0)
endMask = Int32Scalar("endMask", 0)
shrinkAxisMask = Int32Scalar("shrinkAxisMask", 0)

# note output shape is used by expected output's shape
model_output = Output("output", "TENSOR_FLOAT32", "{2}")

model.Operation("STRIDED_SLICE", test_node_input, begins, ends, strides, beginMask, endMask, shrinkAxisMask).To(model_output)

model_input_data = [1, 2, 3, 4]
model_output_data = [2, 3]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
