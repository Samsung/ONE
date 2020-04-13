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

num_input = 3
num_hash = 4
num_bits = 2

model = Model()

hhash = Parameter("hash", "TENSOR_FLOAT32", "{%d, %d}" % (num_hash, num_bits),
                  [0.123, 0.456, -0.321, -0.654, 1.234, 5.678, -4.321, -8.765])
lookup = Input("lookup", "TENSOR_INT32", "{%d, %d}" % (num_input, num_bits))
weight = Input("weight", "TENSOR_FLOAT32", "{%d}" % (num_input))
type_param = Int32Scalar("type_param", 1)  # SPARSE
output = Output("output", "TENSOR_INT32", "{%d}" % (num_hash))
model = model.Operation("LSH_PROJECTION", hhash, lookup, weight,
                        type_param).To(output)

# Omit weight, since this is a sparse projection, for which the optional weight
# input should be left unset.
input0 = {
    lookup: [12345, 54321, 67890, 9876, -12345678, -87654321],
    weight: [],
}

output0 = {output: [1, 2, 2, 0]}

Example((input0, output0))
