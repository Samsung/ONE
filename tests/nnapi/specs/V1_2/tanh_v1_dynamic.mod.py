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

class DynamicInputGenerator:

    def __init__(self, model, model_input_shape_list):
        self.new_shape = 0
        self.model_input = 0
        self.test_input = 0

        # any shape that can be reshaped into model_input_shape
        self.model_input = Input("model_input", "TENSOR_FLOAT32",
                                 self.__getShapeInStr(model_input_shape_list))

        # add Reshape. Output of Reshape dynamic (shapce cannot know at compile time)
        new_shape_str = "{" + str(len(model_input_shape_list)) + "}"
        self.new_shape   = Input("new_shape", "TENSOR_INT32", new_shape_str)

        # shape not known since it is dynamic.. Just use {1}
        # onert should take care of the shape of dynamic tensor
        self.test_input = Internal("internal1", "TENSOR_FLOAT32", "{1}")
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


model = Model()

model_input_shape = [2, 2]

dynamic_layer = DynamicInputGenerator(model, model_input_shape)

test_node_input = dynamic_layer.getTestNodeInput()

# write Tanh test.
# note output shape is dynamic and not known. However, we need to provide any shape
# that has enough memory to hold test output. Otherwise, TestGenerated.cpp will fail
# exact shape of output is not important since onert will calculate the shape at executin time.
# e.g., [4], [1, 4], [2, 2], [4, 1, 1] are OK as long as its element count is 4 (num of input data)
model_output = Output("output", "TENSOR_FLOAT32", "{1, 4}")

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
