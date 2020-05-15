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

input0 = Input("input0", "TENSOR_FLOAT32", "{1, 2, 3}")
multipliers = Input("multipliers", "TENSOR_INT32", "{3}")
output0 = Output("output0", "TENSOR_FLOAT32", "{2, 6, 3}")

model = Model().Operation("TILE", input0, multipliers).To(output0)

input_values = [11, 12, 13,
                21, 22, 23]
multiplier_values = [2, 3, 1]
output_values = [11, 12, 13, 21, 22, 23, 11, 12, 13,
                 21, 22, 23, 11, 12, 13, 21, 22, 23,
                 11, 12, 13, 21, 22, 23, 11, 12, 13,
                 21, 22, 23, 11, 12, 13, 21, 22, 23]

quant8 = DataTypeConverter().Identify({
    input0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
    output0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
})

int32 = DataTypeConverter().Identify({
    input0: ["TENSOR_INT32"],
    output0: ["TENSOR_INT32"],
})

float16 = DataTypeConverter().Identify({
    input0: ["TENSOR_FLOAT16"],
    output0: ["TENSOR_FLOAT16"],
})

Example({
    input0: input_values,
    multipliers: multiplier_values,
    output0: output_values,
}).AddVariations("relaxed", float16, quant8, int32)
