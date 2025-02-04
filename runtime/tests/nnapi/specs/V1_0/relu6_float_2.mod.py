#
# Copyright (C) 2017 The Android Open Source Project
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

# model
model = Model()

d0 = 2
d1 = 26
d2 = 40
d3 = 2

i0 = Input("input", "TENSOR_FLOAT32", "{%d, %d, %d, %d}" % (d0, d1, d2, d3))

output = Output("output", "TENSOR_FLOAT32", "{%d, %d, %d, %d}" % (d0, d1, d2, d3))

model = model.Operation("RELU6", i0).To(output)

# Example 1. Input in operand 0,
rng = d0 * d1 * d2 * d3
input_values = (lambda r = rng: [x * (x % 2 - .5) * .002 for x in range(r)])()
input0 = {i0: input_values}
output_values = [0 if x < 0 else 6 if x > 6 else x for x in input_values]
output0 = {output: output_values}

# Instantiate an example
Example((input0, output0))
