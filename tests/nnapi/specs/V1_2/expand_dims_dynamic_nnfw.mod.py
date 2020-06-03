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
# If axis is from input, the output shape of expand_dims cannot be decided at compilation time.
# Therefore, its output is "dynamic tensor"
#

# case 1
input0 = Input("input0", "TENSOR_FLOAT32", "{2, 2}")
axis0 =  Input("axis0", "TENSOR_INT32", "{}")
output0 = Output("output", "TENSOR_FLOAT32", "{1, 2, 2}")
model0 = Model().Operation("EXPAND_DIMS", input0, axis0).To(output0)

data = [1.2, -3.4, 5.6, 7.8]

Example({
    input0: data,
    axis0: 0,
    output0: data,
}, model = model0, name = "1")

# case 2
output0 = Output("output", "TENSOR_FLOAT32", "{2, 2, 1}")
model0 = Model().Operation("EXPAND_DIMS", input0, axis0).To(output0)

Example({
    input0: data,
    axis0: -1,
    output0: data,
}, model = model0, name = "2")
