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

input0_shape = [4, 1]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, input0_shape, "TENSOR_FLOAT32")

input0 = dynamic_layer.getTestNodeInput() # first input of squeeze is dynamic tensor
squeezeDims = Parameter("squeezeDims", "TENSOR_INT32", "{1}", [1])
output = Output("output", "TENSOR_FLOAT32", "{4}")

model = model.Operation("SQUEEZE", input0, squeezeDims).To(output)

model_input0_data = [1.4, 2.3, 3.2, 4.1]


input0_data = {
    dynamic_layer.getModelInput() : model_input0_data,
    dynamic_layer.getShapeInput() : input0_shape
}

output_data = {
    output : [1.4, 2.3, 3.2, 4.1]
}

# Instantiate an example
Example((input0_data, output_data))
