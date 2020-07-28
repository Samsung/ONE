# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from circle import Model
from circle import ActivationFunctionType
from circle import BuiltinOptions
from circle import BuiltinOperator
from circle import Padding


def assertTrue(cond, msg):
    assert cond, msg


def checkPadding(pad, exp_pad):
    if pad == 'SAME':
        assertTrue(exp_pad == Padding.Padding.SAME, "Padding mismatches")
    elif pad == 'VALID':
        assertTrue(exp_pad == Padding.Padding.VALID, "Padding mismatches")
    else:
        raise SystemExit('Unsupported padding')


def checkActivation(act, exp_act):
    if act == 'relu':
        assertTrue(exp_act == ActivationFunctionType.ActivationFunctionType.RELU,
                   "Activation function mismatches")
    elif act == 'relu6':
        assertTrue(exp_act == ActivationFunctionType.ActivationFunctionType.RELU6,
                   "Activation function mismatches")
    elif act == 'none':
        assertTrue(exp_act == ActivationFunctionType.ActivationFunctionType.NONE,
                   "Activation function mismatches")
    else:
        raise SystemExit('Unsupported activation function')


def checkOpcode(opcode, exp_opcode):
    assertTrue(opcode == exp_opcode,
               "Opcode mismatches (" + str(opcode) + ", " + str(exp_opcode) + ")")


def checkBuiltinOptionType(option, exp_option):
    assertTrue(
        option == exp_option,
        "Built-in option type mismatches (" + str(option) + ", " + str(exp_option) + ")")
