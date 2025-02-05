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

input0_shape = [2, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, input0_shape, "TENSOR_FLOAT32")

input0 = dynamic_layer.getTestNodeInput() # first input of Less is dynamic tensor

# model definition
input1 = Input("input1", "TENSOR_FLOAT32", "{2}")
output = Output("output", "TENSOR_BOOL8", "{2, 2}")

model = model.Operation("LESS", input0, input1).To(output)

model_input0_data = [5, 10]
model_input1_data = [10, 5]

input_data = {
    dynamic_layer.getModelInput() : model_input0_data,
    dynamic_layer.getShapeInput() : input0_shape,

    input1 : model_input1_data
}

output_data = {
    output : [True, False, False, False]
}

# Instantiate an example
Example((input_data, output_data))
