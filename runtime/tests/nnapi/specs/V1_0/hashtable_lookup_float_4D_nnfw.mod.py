#
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

lookups = 4
keys = 3
batchs = 3
cols = 4
rows = 5
features = 2

table = [x for x in range(batchs * cols * rows * features)]
for i in range(batchs):
  for j in range(cols):
    for k in range(rows):
      for l in range(features):
        table[i * cols * rows * features + j * rows * features  + k * features + l] = i * cols * rows * features + j * rows * features  + k * features + l

model = Model()

lookup = Input("lookup", "TENSOR_INT32", "{%d}" % lookups)
key = Input("key", "TENSOR_INT32", "{%d}" % (keys))
value = Input("value", "TENSOR_FLOAT32", "{%d, %d, %d, %d}" % (batchs, cols, rows, features))
output = Output("output", "TENSOR_FLOAT32", "{%d, %d, %d, %d}" % (lookups, cols, rows, features))
hits = Output("hits", "TENSOR_QUANT8_ASYMM", "{%d}, 1.f, 0" % (lookups))
model = model.Operation("HASHTABLE_LOOKUP", lookup, key, value).To([output, hits])

input0 = {lookup:  [1234, -292, -11, 0],
          key: [-11, 0, 1234],
          value: table}

output0 = {output:
           [
               # 2-rd item
               80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
               # Not found
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               # 0-th item
               1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
               # 1-st item
               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
           ],
           hits:
           [
               1, 0, 1, 1,
           ]
}

# Instantiate an example
Example((input0, output0))
