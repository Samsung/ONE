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

input0 = Input("input0", "TENSOR_FLOAT32", "{2, 2}")

output0 = Output("output", "TENSOR_FLOAT32", "{1, 2, 2}")
output1 = Output("output", "TENSOR_FLOAT32", "{2, 1, 2}")
output2 = Output("output", "TENSOR_FLOAT32", "{2, 2, 1}")
output3 = output2

model0 = Model().Operation("EXPAND_DIMS", input0, 0).To(output0)
model1 = Model().Operation("EXPAND_DIMS", input0, 1).To(output1)
model2 = Model().Operation("EXPAND_DIMS", input0, 2).To(output2)
model3 = Model().Operation("EXPAND_DIMS", input0, -1).To(output3)

data = [1.2, -3.4, 5.6, 7.8]

for model, output in [(model0, output0),
                      (model1, output1),
                      (model2, output2),
                      (model3, output3)]:
  quant8 = DataTypeConverter().Identify({
      input0: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
      output: ["TENSOR_QUANT8_ASYMM", 0.5, 127],
  })

  int32 = DataTypeConverter().Identify({
      input0: ["TENSOR_INT32"],
      output: ["TENSOR_INT32"],
  })

  Example({
      input0: data,
      output: data,
  }, model=model).AddVariations("relaxed", quant8, int32, "float16")
