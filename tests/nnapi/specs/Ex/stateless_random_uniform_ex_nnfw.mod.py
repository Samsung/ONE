#
# Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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


model = Model()

i1 = Input("input1", "TENSOR_INT32", "{1}")
i2 = Input("input2", "TENSOR_INT32", "{2}")

o1 = Output("output0", "TENSOR_FLOAT32", "{10}")

model = model.Operation("STATELESS_RANDOM_UNIFORM_EX", i1, i2).To(o1)

# Example.
input0 = {
  i1 : [10],  #input1
  i2 : [1, 1] #input2
}

output0 = {
  o1: [0.09827709, 0.14063823, 0.4553436,
      0.10658443, 0.2075988, 0.30841374,
      0.7489233, 0.90613365, 0.63342273, 
      0.37854457]
}

Example((input0, output0))
