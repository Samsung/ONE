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
base = Input("base", "TENSOR_FLOAT32", "{2, 1}")

exponents = [Input("exponent", "TENSOR_FLOAT32", "{1}"),
             Input("exponent", "TENSOR_FLOAT32", "{1, 2}"),
             Input("exponent", "TENSOR_FLOAT32", "{3, 1, 2}")]

outputs = [Output("output", "TENSOR_FLOAT32", "{2, 1}"),
           Output("output", "TENSOR_FLOAT32", "{2, 2}"),
           Output("output", "TENSOR_FLOAT32", "{3, 2, 2}")]

base_data = [2., 3.]
exponents_data = [[2.],
                  [2., 3.],
                  [0., 0.5, 1., 2., 3., 4.]]

outputs_data = [[4., 9.],
                [4., 8., 9., 27.],
                [1., 2 ** 0.5, 1., 3 ** 0.5, 2., 4., 3., 9., 8., 16., 27., 81.]]

for exponent, output, exponent_data, output_data in zip(exponents, outputs, exponents_data, outputs_data):
    model = Model().Operation("POW", base, exponent).To(output)
    Example({
        base: base_data,
        exponent: exponent_data,
        output: output_data
    }, model=model).AddVariations("relaxed", "float16")
