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

# Adapted from tensorflow/lite/kernels/concatenation_test.cc

input0 = Input("input0", "TENSOR_FLOAT32", "{2, 1, 2}")
input1 = Input("input1", "TENSOR_FLOAT32", "{2, 1, 2}")
input2 = Input("input2", "TENSOR_FLOAT32", "{2, 1, 2}")
input3 = Input("input3", "TENSOR_FLOAT32", "{2, 1, 2}")
axis = 2
output0 = Output("output0", "TENSOR_FLOAT32", "{2, 1, 8}")

model = Model().Operation("CONCATENATION", input0, input1, input2, input3, axis).To(output0)

# FourInputsQuantizedMixedRange
Example({
    input0: [1.0, -3.0, -4.0, -7.0],
    input1: [1.1, 3.1, 4.1, 7.1],
    input2: [1.2, -3.2, -4.2, 7.2],
    input3: [1.3, 3.3, 4.3, 7.3],
    output0: [1.0, -3.0, 1.1, 3.1, 1.2, -3.2, 1.3, 3.3, -4.0, -7.0, 4.1, 7.1, -4.2, 7.2, 4.3, 7.3],
}).AddVariations(DataTypeConverter().Identify({
    input0: ["TENSOR_QUANT8_ASYMM", 0.084, 127],
    input1: ["TENSOR_QUANT8_ASYMM", 0.05, 0],
    input2: ["TENSOR_QUANT8_ASYMM", 0.089, 123],
    input3: ["TENSOR_QUANT8_ASYMM", 0.029, 0],
    output0: ["TENSOR_QUANT8_ASYMM", 0.1, 127],
}), includeDefault=False)

# FourInputsQuantizedMixedRangeClampingLogic
Example({
    input0: [1.0, -3.0, -4.0, -7.0],
    input1: [1.1, 3.1, 4.1, 7.1],
    input2: [1.2, -3.2, -4.2, 7.2],
    input3: [1.3, 3.3, 4.3, 7.3],
    output0: [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
}).AddVariations(DataTypeConverter().Identify({
    input0: ["TENSOR_QUANT8_ASYMM", 0.084, 127],
    input1: ["TENSOR_QUANT8_ASYMM", 0.05, 0],
    input2: ["TENSOR_QUANT8_ASYMM", 0.089, 123],
    input3: ["TENSOR_QUANT8_ASYMM", 0.029, 0],
    output0: ["TENSOR_QUANT8_ASYMM", 0.0078125, 127],
}), includeDefault=False)
