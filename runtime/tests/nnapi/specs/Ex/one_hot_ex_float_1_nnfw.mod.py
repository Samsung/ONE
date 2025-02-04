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

model = Model()


indices = Input("indices", "TENSOR_INT32", "{2, 2}")
depth = Parameter("depth", "TENSOR_INT32", "{1}", [3])
onvalue = Input("onvalue", "TENSOR_FLOAT32", "{1}")
offvalue = Input("offvalue", "TENSOR_FLOAT32", "{1}")

axis0 = Int32Scalar("axis", -1) # default value is -1.
model_output0 = Output("output", "TENSOR_FLOAT32", "{2, 2, 3}")

model0 = model.Operation("ONE_HOT_EX", indices, depth, onvalue, offvalue, axis0).To(model_output0)

model_output_data = ([0., 1., 0.,
                      0., 0., 1.,
                      1., 0., 0.,
                      0., 0., 1.,])

indices_data = [1, 2, 0, 2]
onvalue_data = [1.]
offvalue_data = [0.]

Example(
  {
    indices : indices_data,
    onvalue : onvalue_data,
    offvalue : offvalue_data,

    model_output0 : model_output_data,
  })

