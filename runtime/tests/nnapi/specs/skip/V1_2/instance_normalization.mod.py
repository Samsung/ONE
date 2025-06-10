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

layout = BoolScalar("layout", False) # NHWC

# TEST 1: INSTANCE_NORMALIZATION, gamma = 1, beta = 0, epsilon = 0.0001
i1 = Input("in", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
o1 = Output("out", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
Model().Operation("INSTANCE_NORMALIZATION", i1, 1.0, 0.0, 0.0001, layout).To(o1)

# Instantiate an example
Example({
    i1: [
        0, 1, 0, 2, 0, 2, 0, 4,
        1, -1, -1, 2, -1, -2, 1, 4
    ],
    o1: [
        0.00, -0.499996, 0.00, -0.0999992,
        0.00, -0.0999992, 0.00, 0.6999944,
        0.99995, -0.6999944, -0.99995, 0.499996,
        -0.99995, -1.0999912, 0.99995, 1.2999896
    ]
}).AddNchw(i1, o1, layout).AddVariations("relaxed", "float16")


# TEST 2: INSTANCE_NORMALIZATION, gamma = 2, beta = 10, epsilon = 0.0001
i2 = Input("in", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
o2 = Output("out", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
Model().Operation("INSTANCE_NORMALIZATION", i2, 2.0, 10.0, 0.0001, layout).To(o2)

# Instantiate an example
Example({
    i2: [
        0, 1, 0, 2, 0, 2, 0, 4,
        1, -1, -1, 2, -1, -2, 1, 4
    ],
    o2: [
        10.       ,  9.000008 , 10.       ,  9.8000016, 10.       ,
        9.8000016, 10.       , 11.3999888, 11.9999   ,  8.6000112,
        8.0001   , 10.999992 ,  8.0001   ,  7.8000176, 11.9999   ,
       12.5999792
    ]
}).AddNchw(i2, o2, layout).AddVariations("relaxed", "float16")
