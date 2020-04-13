#
# Copyright (C) 2019 The Android Open Source Project
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

input0 = Input("input0", "TENSOR_FLOAT32", "{3}")
paddings = Parameter("paddings", "TENSOR_INT32", "{1, 2}", [3, 1])
output0 = Output("output0", "TENSOR_FLOAT32", "{7}")

model = Model().Operation("PAD", input0, paddings).To(output0)

Example({
    input0: [1.0, 2.0, 3.0],
    output0: [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
}).AddVariations("float16")

# PAD of TENSOR_FLOAT32 data type is introduced in V1_1.
Example.SetVersion("V1_1", "pad_low_rank")
