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

import dynamic_tensor

model = Model()

input_a_shape = [1, 2, 2]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, input_a_shape, "TENSOR_FLOAT32")

input_a = dynamic_layer.getTestNodeInput() # first input of select_v2 is dynamic tensor

# model definition
input_cond = Input("input_cond", "TENSOR_BOOL8", "{1, 2}")
input_b = Input("input_b", "TENSOR_FLOAT32", "{1, 2, 1}")
output = Output("output", "TENSOR_FLOAT32", "{1, 2, 2}")

model = model.Operation("SELECT_V2_EX", input_cond, input_a, input_b).To(output)

model_input_cond_data = [False, True]
model_input_a_data = [1, 2 ,3 ,4]
model_input_b_data = [5, 6]

input_data = {
    input_cond : model_input_cond_data,
    dynamic_layer.getModelInput() : model_input_a_data,
    dynamic_layer.getShapeInput() : input_a_shape,
    input_b : model_input_b_data
}

output_data = {
    output : [5, 2, 6, 4]
}

# Instantiate an example
Example((input_data, output_data))
