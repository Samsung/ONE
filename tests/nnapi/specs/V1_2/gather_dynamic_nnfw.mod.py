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
Testing Gather op when the input1 is dynamic.
       input1 [1, 2, 3, 4]  shape [4]  (value of shape will be [1, 2, 3, 4])
          |             |
          +-------------+
          |
       Reshape (added by DynamicInputGenerator since it generates its output to be dynamic)
          |
          |        axis = 0       input2 [2]
          |             |             |
          +-------------+-------------+
          |
          |
          | dynamic tensor at compilation time but the shape will be [2, 2, 3, 4] at execution time
          |
         Gather
          |
        output (dynamic tensor, [2, 2, 3, 4] at execution time)
'''
import dynamic_tensor

model = Model()

input1_shape = [1, 2, 3, 4]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, input1_shape, "TENSOR_FLOAT32")

node_input = dynamic_layer.getTestNodeInput()

input2 = Input("intput2", "TENSOR_INT32", "{2}")
axis = Int32Scalar("axis", 0)
output = Output("output", "TENSOR_FLOAT32", "{2,2,3,4}")
model = model.Operation("GATHER", node_input, axis, input2).To(output)

input1_data = [1.123456789123456789, 2.123456789123456789, 3.123456789123456789, 4.123456789123456789,
               5.123456789123456789, 6.123456789123456789, 7.123456789123456789, 8.123456789123456789,
               9.123456789123456789, 10.123456789123456789, 11.123456789123456789, 12.123456789123456789,
               13.123456789123456789, 14.123456789123456789, 15.123456789123456789, 16.123456789123456789,
               17.123456789123456789, 18.123456789123456789, 19.123456789123456789, 20.123456789123456789,
               21.123456789123456789, 22.123456789123456789, 23.123456789123456789, 24.123456789123456789
               ]

input0 = {
          dynamic_layer.getModelInput() : input1_data, # input 1
          dynamic_layer.getShapeInput() : input1_shape,

          input2 : [0, 0]  # input 2
          }

output0 = {
          output: # output
          [1.123456789123456789, 2.123456789123456789, 3.123456789123456789, 4.123456789123456789,
          5.123456789123456789, 6.123456789123456789, 7.123456789123456789, 8.123456789123456789,
          9.123456789123456789, 10.123456789123456789, 11.123456789123456789, 12.123456789123456789,
          13.123456789123456789, 14.123456789123456789, 15.123456789123456789, 16.123456789123456789,
          17.123456789123456789, 18.123456789123456789, 19.123456789123456789, 20.123456789123456789,
          21.123456789123456789, 22.123456789123456789, 23.123456789123456789, 24.123456789123456789,
          1.123456789123456789, 2.123456789123456789, 3.123456789123456789, 4.123456789123456789,
          5.123456789123456789, 6.123456789123456789, 7.123456789123456789, 8.123456789123456789,
          9.123456789123456789, 10.123456789123456789, 11.123456789123456789, 12.123456789123456789,
          13.123456789123456789, 14.123456789123456789, 15.123456789123456789, 16.123456789123456789,
          17.123456789123456789, 18.123456789123456789, 19.123456789123456789, 20.123456789123456789,
          21.123456789123456789, 22.123456789123456789, 23.123456789123456789, 24.123456789123456789]
          }

# Instantiate an example
Example((input0, output0))
