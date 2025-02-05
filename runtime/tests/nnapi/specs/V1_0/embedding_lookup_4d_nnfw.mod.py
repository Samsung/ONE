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

lookups = 3
N = 5
H = 2
W = 4
C = 2

actual_values = [x for x in range(N * H * W * C)]

model = Model()
index = Input("index", "TENSOR_INT32", "{%d}"%lookups)
value = Input("value", "TENSOR_INT32", "{%d, %d, %d, %d}" % (N, H, W, C))
output = Output("output", "TENSOR_INT32", "{%d, %d, %d, %d}" % (lookups, H, W, C))
model = model.Operation("EMBEDDING_LOOKUP", index, value).To(output)

input0 = {index: [4, 0, 2],
          value: actual_values}

output0 = {output:
           [
               64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, # Row 4
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,           # Row 0
               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, # Row 2
           ]}

# Instantiate an example
Example((input0, output0))
