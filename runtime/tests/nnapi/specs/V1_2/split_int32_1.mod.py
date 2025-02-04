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

# model
input0 = Input("input0", "TENSOR_INT32", "{6}")
axis = Int32Scalar("axis", 0)
num_splits = Int32Scalar("num_splits", 3)
output0 = Output("output0", "TENSOR_INT32", "{2}")
output1 = Output("output1", "TENSOR_INT32", "{2}")
output2 = Output("output2", "TENSOR_INT32", "{2}")

model = Model().Operation("SPLIT", input0, axis, num_splits).To((output0, output1, output2))

# Example 1.
input_dict = {
    input0: [1, 2, 3, 4, 5, 6]
}
output_dict = {
    output0: [1, 2],
    output1: [3, 4],
    output2: [5, 6],
}

# Instantiate an example
Example((input_dict, output_dict)).AddRelaxed()
