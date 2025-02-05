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

num_input = 3
num_hash = 4
num_bits = 2

model = Model()

hhash = Parameter("hash", "TENSOR_FLOAT16", "{%d, %d}" % (num_hash, num_bits),
                  [0.123, 0.456, -0.321, -0.654, 1.234, 5.678, -4.321, -8.765])
lookup = Input("lookup", "TENSOR_INT32", "{%d, %d}" % (num_input, num_bits))
weight = Input("weight", "TENSOR_FLOAT16", "{%d}" % (num_input))
type_param = Int32Scalar("type_param", 2)  # DENSE
output = Output("output", "TENSOR_INT32", "{%d}" % (num_hash * num_bits))
model = model.Operation("LSH_PROJECTION", hhash, lookup, weight,
                        type_param).To(output)

#TODO: weight should be a constant, too.
input0 = {
    lookup: [12345, 54321, 67890, 9876, -12345678, -87654321],
    weight: [0.12, 0.34, 0.56]
}
output0 = {output: [1, 1, 1, 1, 1, 0, 0, 0]}

Example((input0, output0)).AddVariations("float16");
