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

# please refer to tanh_v1_dynamic.mod.py to see how to write test about dynamic tensor
import dynamic_tensor

model = Model()

input1_shape = [2, 3]
dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, input1_shape, "TENSOR_FLOAT32")

input1 = dynamic_layer.getTestNodeInput()
input2 = Input("op2", "TENSOR_FLOAT32", "{2, 3}")
axis0 = Int32Scalar("axis0", 0)
model_output = Output("output", "TENSOR_FLOAT32", "{4, 3}")

model =model.Operation("CONCATENATION", input1, input2, axis0).To(model_output)

input1_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
input2_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
model_output_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

input_list = {
    dynamic_layer.getModelInput(): input1_data,
    dynamic_layer.getShapeInput() : input1_shape,

    input2 : input2_data,
    }

output_list= {
    model_output: model_output_data
    }

Example((input_list, output_list))
