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

#
# If multiplier is from input, the output shape of tile cannot be decided at compilation time.
# Therefore, its output is "dynamic tensor"
#

input0 = Input("input0", "TENSOR_FLOAT32", "{3}")
multi0 =  Input("input1", "TENSOR_INT32", "{}")
output0 = Output("output", "TENSOR_FLOAT32", "{6}")

model0 = Model().Operation("TILE", input0, multi0).To(output0)

Example({
    input0: [1.2, -3.4, 5.6],
    multi0: [2],
    output0: [1.2, -3.4, 5.6, 1.2, -3.4, 5.6],
})
