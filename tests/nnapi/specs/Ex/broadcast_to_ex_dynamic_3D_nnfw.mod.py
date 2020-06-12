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
# This op(broadcast_to)'s output shape is depend for entered shape,
# So the output shape of broadcast_to cannot be decided at compilation time.
# Therefore, its output is "dynamic tensor"
#

# model

import dynamic_tensor

model = Model()

i1_shape = [2, 1, 2]
i1_data = [1, 2, 3, 4]

i2_data = [2, 3, 2]

o1_data = [1, 2,
           1, 2,
           1, 2,

           3, 4,
           3, 4,
           3, 4,]

dynamic_layer = dynamic_tensor.DynamicInputGenerator(model, i1_shape, "TENSOR_INT32")

i1 = dynamic_layer.getTestNodeInput()
i2 = Input("input2", "TENSOR_INT32", "{3}")
o1 = Output("output0", "TENSOR_INT32", "{2, 3, 2}")

model = model.Operation("BROADCAST_TO_EX", i1, i2).To(o1)

# Example.
input0 = {
  dynamic_layer.getModelInput(): i1_data,
  dynamic_layer.getShapeInput(): i1_shape,

  i2: i2_data
}

output0 = {
  o1: o1_data
}

Example((input0, output0))
