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


import dynamic_tensor

model = Model()

model_input_shape = [2, 1, 2, 2]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, model_input_shape, "TENSOR_BOOL8")

test_node_input = dynamic_layer.getTestNodeInput()

# model
i2 = Input("op2", "TENSOR_BOOL8", "{2, 1, 2, 2}")
t1 = Internal("op3", "TENSOR_BOOL8", "{}") # result of first LOGICAL_OR
act = Int32Scalar("act", 0) # scalar activation
o1 = Output("op3", "TENSOR_BOOL8", "{2, 1, 2, 2}")

model = model.Operation("LOGICAL_OR", test_node_input, i2, act).To(t1) # first add
model = model.Operation("LOGICAL_OR", t1, i2, act).To(o1)              # second add

model_input1_data = [False, False, True, True, False, True, False, True]
model_input2_data = [False, True, False, True, False, False, True, True]
model_output_data = [False, True, True, True, False, True, True, True]

Example({
    dynamic_layer.getModelInput(): model_input1_data,
    dynamic_layer.getShapeInput(): model_input1_shape,
    i2 : model_input2_data,

    model_output: model_output_data,
})
