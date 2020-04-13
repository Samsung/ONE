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

i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2, 2, 3}") # input 0
o1 = Output("op2", "TENSOR_FLOAT32", "{2, 2, 2, 3}") # output 0
axis = Int32Scalar("axis", -1) # last axis

quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.1, 32),
    o1: ("TENSOR_QUANT8_ASYMM", 1.0 / 128, 128)
})

example0 = {
    i1: [ 0,  3,  4,
          3,  0,  4,
          8,  6,  0,
         12,  0,  9,
          9, 12, 20,
         12, 15, 16,
         20,  9, 12,
         16, 15, 12],
    o1: [0.00, 0.60, 0.80,
         0.60, 0.00, 0.80,
         0.80, 0.60, 0.00,
         0.80, 0.00, 0.60,
         0.36, 0.48, 0.80,
         0.48, 0.60, 0.64,
         0.80, 0.36, 0.48,
         0.64, 0.60, 0.48]
}

# All dimensions other than 4, without axis parameter
Model().Operation("L2_NORMALIZATION", i1).To(o1)
Example(example0).AddRelaxed().AddAllDims(i1, o1).AddVariations("relaxed", "float16", quant8)

# L2_NORMALIZATION of rank 4 is introduced in V1_0.
Example.SetVersion("V1_0", "l2_normalization_v1_2_dim4_axis3")
