#
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
# Copyright (C) 2020 The Android Open Source Project
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

i1 = Input("input0", "TENSOR_INT32", "{3, 1, 2}")
i2 = Input("input1", "TENSOR_INT32", "{3, 1, 2}")
i3 = Output("output0", "TENSOR_INT32", "{3, 1, 2}")

model = Model().Operation("MINIMUM", i1, i2).To(i3)

input0 = {i1:
          [129, 12, 15, 130, -77, 33],
          i2:
          [44, 127, -25, 5, 39, 27]}

output0 = {i3:
           [44, 12, -25, 5, -77, 27]}

Example((input0, output0))
