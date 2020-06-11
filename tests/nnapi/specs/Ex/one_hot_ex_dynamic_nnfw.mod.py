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

#
# If depth or axis is from input, the output shape of one_hot cannot be decided at compilation time.
# Therefore, its output is "dynamic tensor"
#

import dynamic_tensor

model = Model()

indice_shape = [2, 2]
indice_input = [1, 2, 0, 2]

depth = Int32Scalar("depth", 3)
onvalue = Float32Scalar("on", 1.) # default value is 1.
offvalue = Float32Scalar("off", 0.) # default value is 0.
axis0 = Int32Scalar("axis", -1) # default value is -1.
model_output0 = Output("output", "TENSOR_FLOAT32", "{2, 2, 3}")

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, indice_shape, "TENSOR_INT32")
node_input = dynamic_layer.getTestNodeInput()

model0 = model.Operation("ONE_HOT_EX", node_input, depth, onvalue, offvalue, axis0).To(model_output0)

model_output_data = ([0., 1., 0.,
                      0., 0., 1.,
                      1., 0., 0.,
                      0., 0., 1.,])

Example(
  {
    dynamic_layer.getModelInput() : indice_input,
    dynamic_layer.getShapeInput() : indice_shape,

    model_output0 : model_output_data,
  })

