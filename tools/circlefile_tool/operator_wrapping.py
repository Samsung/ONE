#!/usr/bin/python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tflite.Operator
import tflite.OperatorCode
import tflite.BuiltinOperator
import tflite.ActivationFunctionType
from operation import Operation


# Match enum value integer to name string
# Assumption 1: enum value is defined by old style (can be used on python 2)
# Assumption 2: when class define enum value, only constant value is defined and methods are not defined
# Assumption 3: only integer value is set by constant definition
def BuildEnumClassStrMap(obj):
    ret = {}
    for fieldName in dir(obj):
        if (not fieldName.startswith('_')):
            fieldValue = getattr(obj, fieldName)
            if (isinstance(fieldValue, (int))):
                ret[fieldValue] = fieldName
    return ret


class EnumStrMaps():
    BuiltinOpcode = BuildEnumClassStrMap(tflite.BuiltinOperator.BuiltinOperator())
    ActivationFunctionType = BuildEnumClassStrMap(
        tflite.ActivationFunctionType.ActivationFunctionType())
    BuiltinOptions = BuildEnumClassStrMap(tflite.BuiltinOptions.BuiltinOptions())


def GetAttribute(o, *args):
    import functools
    return functools.reduce(getattr, args, o)


def BuildBuiltinOptionGen():
    bo_gen = {}
    for val_enum in EnumStrMaps.BuiltinOptions:
        val_str = EnumStrMaps.BuiltinOptions[val_enum]
        try:
            # Dynamically import Builtin Option classes
            # 0 (NONE) is the only exception that does not have no corresponding flatbuffer-generated class
            module = __import__("tflite." + val_str)
            bo_gen[val_enum] = GetAttribute(module, val_str, val_str)
        except ImportError as e:
            assert val_enum == 0 and val_str == "NONE"
    return bo_gen


class OptionLoader:
    builtinOptionGen = BuildBuiltinOptionGen()

    @staticmethod
    def GetBuiltinOptions(options_type, options_table):
        if (options_table == None) and (options_type != 0):
            print(
                "Bad flatbuffer file: undefined builtin option table with defined option type"
            )
            exit(1)
        options = OptionLoader.builtinOptionGen[options_type]()
        options.Init(options_table.Bytes, options_table.Pos)
        return options


class Operator(object):
    def __init__(self, operator_idx, tf_operator, input_tensors, output_tensors,
                 opcode_str):
        self.operator_idx = operator_idx
        self.tf_operator = tf_operator
        self.inputs = input_tensors
        self.outputs = output_tensors
        self.opcode_str = opcode_str
        self.operation = Operation(self.tf_operator, self.opcode_str, self.inputs,
                                   self.outputs)
        self.fused_activation = "NONE"
        self.SetupBuiltinOption()
        self.SetupFusedActivation()

    def SetupBuiltinOption(self):
        try:
            self.options = OptionLoader.GetBuiltinOptions(
                self.tf_operator.BuiltinOptionsType(), self.tf_operator.BuiltinOptions())
        except KeyError:
            self.options = 0
            return

    def SetupFusedActivation(self):
        # FIXME: workaround for ops such as custom
        try:
            options = OptionLoader.GetBuiltinOptions(
                self.tf_operator.BuiltinOptionsType(), self.tf_operator.BuiltinOptions())
        except KeyError:
            return

        # fused activation function
        try:
            activation_code = options.FusedActivationFunction()
            self.fused_activation = EnumStrMaps.ActivationFunctionType[activation_code]
        except AttributeError:
            # This operator does not support FusedActivationFunction
            pass
