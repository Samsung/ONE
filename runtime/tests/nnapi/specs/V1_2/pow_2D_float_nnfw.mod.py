#
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
base = Input("base", "TENSOR_FLOAT32", "{2, 2}")
exponent = Input("exponent", "TENSOR_FLOAT32", "{2, 2}")
output = Output("output",   "TENSOR_FLOAT32", "{2, 2}")

base_data = [2., 9., 4., 5.]
exponent_data = [1., 0.5, 2., 3.]
output_data = [2., 3., 16. , 125.]

model = Model().Operation("POW", base, exponent).To(output)
Example({
    base: base_data,
    exponent: exponent_data,
    output: output_data
}, model=model)
