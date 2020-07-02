#
# Copyright (C) 2018 The Android Open Source Project
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

#
# This op(broadcast_to)'s output shape is depend for entered shape,
# So the output shape of broadcast_to cannot be decided at compilation time.
# Therefore, its output is "dynamic tensor"
#

# model

model = Model()

i1 = Input("input1", "TENSOR_INT32", "{3}")
i2 = Parameter("input2", "TENSOR_INT32", "{2}", [3, 3])
o1 = Output("output0", "TENSOR_INT32", "{3, 3}")

model = model.Operation("BROADCAST_TO_EX", i1, i2).To(o1)

# Example.
input0 = {
  i1: [1, 2, 3], #input 0
}

output0 = {
  o1: [1, 2, 3,
       1, 2, 3,
       1, 2, 3]
}

Example((input0, output0))
