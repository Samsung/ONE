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

import numpy

input0 = Input("input0", "TENSOR_QUANT8_ASYMM", "{1, 1, 2, 3}, 2.3, 4")
paddings = Parameter("paddings", "TENSOR_INT32", "{4, 2}", [1, 2,
                                                            3, 4,
                                                            3, 3,
                                                            2, 1])
pad_value = Int32Scalar("pad_value", 3)
output0 = Output("output0", "TENSOR_QUANT8_ASYMM", "{4, 8, 8, 6}, 2.3, 4")

model = Model().Operation("PAD_V2", input0, paddings, pad_value).To(output0)

Example({
    input0: [1, 2, 3,
             4, 5, 6],
    output0: np.pad([[[[1, 2, 3],
                       [4, 5, 6]]]],
                    [[1, 2],
                     [3, 4],
                     [3, 3],
                     [2, 1]],
                    "constant",
                    constant_values=3).flatten().tolist(),
})
