#
# Copyright (C) 2018 The Android Open Source Project
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

model_input_shape = [2, 3]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape)

test_node_input = dynamic_layer.getTestNodeInput()

model_output = Output("output", "TENSOR_FLOAT32", "{2, 3}")

model.Operation("FILL_EX", test_node_input).To(model_output)

model_input_data = [50.0, 10.0, 1.5, 0.2, 0.777, 0.6]

Example({
    # sue these two as input
    dynamic_layer.getModelInput() : model_input_data,
    dynamic_layer.getShapeInput() : model_input_shape,

    model_output : model_input_shape,
})