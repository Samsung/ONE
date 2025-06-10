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

input0 = Input("input0", "TENSOR_QUANT8_ASYMM", "{1, 2, 3, 1}, 2.3, 0")
paddings = Parameter("paddings", "TENSOR_INT32", "{4, 2}", [0, 0,
                                                            0, 2,
                                                            1, 3,
                                                            0, 0])
output0 = Output("output0", "TENSOR_QUANT8_ASYMM", "{1, 4, 7, 1}, 2.3, 0")

model = Model().IntroducedIn("V1_1").Operation("PAD", input0, paddings).To(output0)

Example({
    input0: [1, 2, 3,
             4, 5, 6],
    output0: [0, 1, 2, 3, 0, 0, 0,
              0, 4, 5, 6, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0],
})
