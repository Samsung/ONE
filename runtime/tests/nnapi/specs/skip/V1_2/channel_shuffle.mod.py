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

i1 = Input("op1", "TENSOR_FLOAT32", "{2, 2, 3, 12}") # input 0
o1 = Output("op2", "TENSOR_FLOAT32", "{2, 2, 3, 12}") # output 0
axis = Int32Scalar("axis", -1) # last axis
Model().Operation("CHANNEL_SHUFFLE", i1, 3, axis).To(o1)

# Additional data type
quant8 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM", 0.25, 128),
    o1: ("TENSOR_QUANT8_ASYMM", 0.25, 128)
})

Example({
    i1: list(range(2*2*3*12)),
    o1: [  0,   4,   8,   1,   5,   9,   2,   6,  10,   3,   7,  11,
          12,  16,  20,  13,  17,  21,  14,  18,  22,  15,  19,  23,
          24,  28,  32,  25,  29,  33,  26,  30,  34,  27,  31,  35,
          36,  40,  44,  37,  41,  45,  38,  42,  46,  39,  43,  47,
          48,  52,  56,  49,  53,  57,  50,  54,  58,  51,  55,  59,
          60,  64,  68,  61,  65,  69,  62,  66,  70,  63,  67,  71,
          72,  76,  80,  73,  77,  81,  74,  78,  82,  75,  79,  83,
          84,  88,  92,  85,  89,  93,  86,  90,  94,  87,  91,  95,
          96, 100, 104,  97, 101, 105,  98, 102, 106,  99, 103, 107,
         108, 112, 116, 109, 113, 117, 110, 114, 118, 111, 115, 119,
         120, 124, 128, 121, 125, 129, 122, 126, 130, 123, 127, 131,
         132, 136, 140, 133, 137, 141, 134, 138, 142, 135, 139, 143]
}).AddVariations("relaxed", quant8, "float16").AddAllDimsAndAxis(i1, o1, axis)
