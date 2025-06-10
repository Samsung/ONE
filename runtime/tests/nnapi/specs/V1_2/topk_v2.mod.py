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
import collections

TestCase = collections.namedtuple("TestCase", [
    "inp", "inp_data", "k", "out_values", "out_values_data", "out_indices",
    "out_indices_data"
])

test_cases = [
    TestCase(
        inp=Input("input", "TENSOR_FLOAT32", "{2, 2}"),
        inp_data=[-2.0, 0.2, 0.8, 0.1],
        k=Int32Scalar("k", 2),
        out_values=Output("out_values", "TENSOR_FLOAT32", "{2, 2}"),
        out_values_data=[0.2, -2.0, 0.8, 0.1],
        out_indices=Output("out_indices", "TENSOR_INT32", "{2, 2}"),
        out_indices_data=[1, 0, 0, 1]),
    TestCase(
        inp=Input("input", "TENSOR_FLOAT32", "{2, 3}"),
        inp_data=[-2.0, -3.0, 0.2, 0.8, 0.1, -0.1],
        k=Int32Scalar("k", 2),
        out_values=Output("out_values", "TENSOR_FLOAT32", "{2, 2}"),
        out_values_data=[0.2, -2.0, 0.8, 0.1],
        out_indices=Output("out_indices", "TENSOR_INT32", "{2, 2}"),
        out_indices_data=[2, 0, 0, 1]),
    TestCase(
        inp=Input("input", "TENSOR_FLOAT32", "{2, 4}"),
        inp_data=[-2.0, -3.0, -4.0, 0.2, 0.8, 0.1, -0.1, -0.8],
        k=Int32Scalar("k", 2),
        out_values=Output("out_values", "TENSOR_FLOAT32", "{2, 2}"),
        out_values_data=[0.2, -2.0, 0.8, 0.1],
        out_indices=Output("out_indices", "TENSOR_INT32", "{2, 2}"),
        out_indices_data=[3, 0, 0, 1]),
    TestCase(
        inp=Input("input", "TENSOR_FLOAT32", "{8}"),
        inp_data=[-2.0, -3.0, -4.0, 0.2, 0.8, 0.1, -0.1, -0.8],
        k=Int32Scalar("k", 2),
        out_values=Output("out_values", "TENSOR_FLOAT32", "{2}"),
        out_values_data=[0.8, 0.2],
        out_indices=Output("out_indices", "TENSOR_INT32", "{2}"),
        out_indices_data=[4, 3]),
    TestCase(
        inp=Input("input", "TENSOR_QUANT8_ASYMM", "{2, 3}, 2.0, 128"),
        inp_data=[1, 2, 3, 251, 250, 249],
        k=Int32Scalar("k", 2),
        out_values=Output("out_values", "TENSOR_QUANT8_ASYMM", "{2, 2}, 2.0, 128"),
        out_values_data=[3, 2, 251, 250],
        out_indices=Output("out_indices", "TENSOR_INT32", "{2, 2}"),
        out_indices_data=[2, 1, 0, 1]),
    TestCase(
        inp=Input("input", "TENSOR_INT32", "{2, 3}"),
        inp_data=[1, 2, 3, 10251, 10250, 10249],
        k=Int32Scalar("k", 2),
        out_values=Output("out_values", "TENSOR_INT32", "{2, 2}"),
        out_values_data=[3, 2, 10251, 10250],
        out_indices=Output("out_indices", "TENSOR_INT32", "{2, 2}"),
        out_indices_data=[2, 1, 0, 1]),
]

for test_case in test_cases:
  model = Model().Operation("TOPK_V2", test_case.inp, test_case.k).To(
      test_case.out_values, test_case.out_indices)
  Example({
      test_case.inp: test_case.inp_data,
      test_case.out_values: test_case.out_values_data,
      test_case.out_indices: test_case.out_indices_data
  }, model=model).AddVariations("relaxed", "float16")
