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
class to add dynamic tensor input by adding Reshape layer

Example test code would look like the following:

--------
import dynamic_tensor

model = Model()

model_input_shape = [1, 2, 3]

dynamic_layer = DynamicInputGenerator(model, model_input_shape)

test_node_input = dynamic_layer.getTestNodeInput()

# write ABS test. input is `test_input`
# note output shape is dynamic and not known.
# However, we need to provide any shape that can hold test output. Otherwise, TestGenerated.cpp will fail
model_output = Output("output", "TENSOR_FLOAT32", "{1, 6}")

model.Operation("ABS", test_node_input).To(model_output)

model_input_data = [1, -2, 3, -4, 5, -6]
model_output_data = [1, 2, 3, 4, 5, 6]

Example({
    # use these two as input
    dynamic_layer.getModelInput(): model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output: model_output_data,
})
--------

The above code creates a model with two inputs and the values in the first input (dynamic tensor)
is passed to the input of testing op.

      input [1, 2, 3]  shape [3]  (value of shape will be [1, 2, 3])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          | dynamic tensor at compilation time but the shape will be [1, 2, 3] at execution time
          |
         Abs
          | dynamic tensor, [1, 2, 3] at execution time
'''

from test_generator import Input
from test_generator import Internal

class DynamicInputGenerator:

    def __init__(self, model, model_input_shape_list, tensor_type='TENSOR_FLOAT32'):
        self.new_shape = 0
        self.model_input = 0
        self.test_input = 0

        # any shape that can be reshaped into model_input_shape
        self.model_input = Input("model_input", tensor_type,
                                 self.__getShapeInStr(model_input_shape_list))

        # add Reshape to make input of Abs dynamic
        new_shape_str = "{" + str(len(model_input_shape_list)) + "}"
        self.new_shape   = Input("new_shape", "TENSOR_INT32", new_shape_str)

        # shape not known since it is dynamic. Use a scalar {} just like TFL Converter.
        self.test_input = Internal("internal1", tensor_type, "{}")
        model.Operation("RESHAPE", self.model_input, self.new_shape).To(self.test_input)

    # convert, e.g., [1, 2, 3] to "{1, 2, 3}"
    def __getShapeInStr(self, shape_list):
        str_shape = ""
        i = 0
        for dim in shape_list:
            if i == 0:
                str_shape = "{" + str(dim)
            else:
                str_shape = str_shape + ", " + str(dim)
            i += 1
        str_shape = str_shape + "}"
        return str_shape

    def getTestNodeInput(self):
        return self.test_input

    def getShapeInput(self):
        return self.new_shape

    def getModelInput(self):
        return self.model_input
