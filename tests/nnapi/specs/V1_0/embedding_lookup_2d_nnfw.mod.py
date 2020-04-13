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
rows = 3
columns = 2

actual_values = [x for x in range(rows * columns)]
for i in range(rows):
  for j in range(columns):
    actual_values[(i * columns + j)] = i + j / 10.

model = Model()
index = Input("index", "TENSOR_INT32", "{%d}"%lookups)
value = Input("value", "TENSOR_FLOAT32", "{%d, %d}" % (rows, columns))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d}" % (lookups, columns))
model = model.Operation("EMBEDDING_LOOKUP", index, value).To(output)

input0 = {index: [1, 0, 2],
          value: actual_values}

output0 = {output:
           [
               1.0, 1.1, # Row 1
               0.0, 0.1, # Row 0
               2.0, 2.1, # Row 2
           ]}

# Instantiate an example
Example((input0, output0))
