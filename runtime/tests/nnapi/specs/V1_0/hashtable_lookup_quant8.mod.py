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

lookups = 4
keys = 3
rows = 3
features = 2

table = [x for x in range(rows * features)]
for i in range(rows):
  for j in range(features):
    table[i * features + j] = i * 10 + j

model = Model()

lookup = Input("lookup", "TENSOR_INT32", "{%d}" % (lookups))
key = Input("key", "TENSOR_INT32", "{%d}" % (keys))
value = Input("value", "TENSOR_QUANT8_ASYMM", "{%d, %d}, 0.5f, 0" % (rows, features))
output = Output("output", "TENSOR_QUANT8_ASYMM", "{%d, %d}, 0.5f, 0" % (lookups, features))
hits = Output("hits", "TENSOR_QUANT8_ASYMM", "{%d}, 1.f, 0" % (lookups))
model = model.Operation("HASHTABLE_LOOKUP", lookup, key, value).To([output, hits])

input0 = {lookup:  [123, 250, 255, 0],
          key: [0, 123, 255],
          value: table}

output0 = {output:
           [
               10, 11, # 2nd item
               0, 0,   # Not found
               20, 21, # 1st item
               0, 1,   # 0th item
           ],
           hits:
           [
               1, 0, 1, 1,
           ]
}

# Instantiate an example
Example((input0, output0))
