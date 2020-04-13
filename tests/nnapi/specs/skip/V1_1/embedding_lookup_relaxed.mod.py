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

lookups = 3
rows = 3
columns = 2
features = 4

actual_values = [x for x in range(rows * columns * features)]
for i in range(rows):
  for j in range(columns):
    for k in range(features):
      actual_values[(i * columns + j) * features + k] = i + j / 10. + k / 100.

model = Model()
index = Input("index", "TENSOR_INT32", "{%d}"%lookups)
value = Input("value", "TENSOR_FLOAT32", "{%d, %d, %d}" % (rows, columns, features))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d, %d}" % (lookups, columns, features))
model = model.Operation("EMBEDDING_LOOKUP", index, value).To(output)
model = model.RelaxedExecution(True)

input0 = {index: [1, 0, 2],
          value: actual_values}

output0 = {output:
           [
               1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  # Row 1
               0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  # Row 0
               2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  # Row 2
           ]}

# Instantiate an example
Example((input0, output0))
