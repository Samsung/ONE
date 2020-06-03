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

import itertools
import collections

Operand = collections.namedtuple(
    "Operand", ["name", "as_input", "as_output", "data", "supports_relaxation"])

operands = [
    Operand(
        name="float16",
        as_input=Input("input0", "TENSOR_FLOAT16", "{2, 3}"),
        as_output=Output("output0", "TENSOR_FLOAT16", "{2, 3}"),
        data=[1, 2, 3, 4, 5, 6],
        supports_relaxation=False),
    Operand(
        name="float32",
        as_input=Input("input0", "TENSOR_FLOAT32", "{2, 3}"),
        as_output=Output("output0", "TENSOR_FLOAT32", "{2, 3}"),
        data=[1, 2, 3, 4, 5, 6],
        supports_relaxation=True),
    Operand(
        name="int32",
        as_input=Input("input0", "TENSOR_INT32", "{2, 3}"),
        as_output=Output("output0", "TENSOR_INT32", "{2, 3}"),
        data=[1, 2, 3, 4, 5, 6],
        supports_relaxation=False),
    Operand(
        name="quant8",
        as_input=Input("input0", "TENSOR_QUANT8_ASYMM", "{2, 3}, 4.0, 100"),
        as_output=Output("output0", "TENSOR_QUANT8_ASYMM", "{2, 3}, 4.0, 100"),
        data=[1, 2, 3, 4, 5, 6],
        supports_relaxation=False),
]

for operand1, operand2 in itertools.product(operands, operands):
  input0 = operand1.as_input
  output0 = operand2.as_output

  model = Model().Operation("CAST", input0).To(output0)

  example = Example({
      input0: operand1.data,
      output0: operand2.data,
  }, model=model, name='{}_to_{}'.format(operand1.name, operand2.name))

  if operand1.supports_relaxation or operand2.supports_relaxation:
    example.AddRelaxed()


# Test overflow and underflow.
operands = [
    Operand(
        name="float16",
        as_input=Input("input0", "TENSOR_FLOAT16", "{2}"),
        as_output=None,
        data=[-1, 256],
        supports_relaxation=False),
    Operand(
        name="float32",
        as_input=Input("input0", "TENSOR_FLOAT32", "{2}"),
        as_output=None,
        data=[-1, 256],
        supports_relaxation=True),
    Operand(
        name="int32",
        as_input=Input("input0", "TENSOR_INT32", "{2}"),
        as_output=None,
        data=[-1, 256],
        supports_relaxation=False),
]

for operand1 in operands:
  input0 = operand1.as_input
  output0 = Output("output0", "TENSOR_QUANT8_ASYMM", "{2}, 4.0, 100")

  model = Model().Operation("CAST", input0).To(output0)

  example = Example({
      input0: operand1.data,
      output0: [0, 255],
  }, model=model, name='{}_to_quant8_overflow'.format(operand1.name))

  if operand1.supports_relaxation:
    example.AddRelaxed()
