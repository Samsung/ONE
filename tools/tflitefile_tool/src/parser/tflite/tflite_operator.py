#!/usr/bin/python

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

from ir.operator import Operator
from .tflite_enum_str_maps import EnumStrMaps
from .tflite_option import OptionLoader, GetStringOptions


class TFLiteOperator(Operator):
    def __init__(self, operator_idx, tf_operator, input_tensors, output_tensors,
                 opcode_str):
        super(TFLiteOperator, self).__init__()

        self.index = operator_idx
        self.inputs = input_tensors
        self.outputs = output_tensors
        self.op_name = opcode_str
        self.activation = "NONE"
        self.options = ""

        self.tf_operator = tf_operator
        self.tf_options = None
        self.SetupBuiltinOption()
        self.SetupFusedActivation()

    def SetupBuiltinOption(self):
        # FIXME: workaround for ops such as custom
        try:
            self.tf_options = OptionLoader.GetBuiltinOptions(
                self.tf_operator.BuiltinOptionsType(), self.tf_operator.BuiltinOptions())
            if self.tf_options == None:
                return

            option_str = GetStringOptions(self.op_name, self.tf_options)
            if option_str is None:
                return

            self.options = option_str
        except KeyError:
            return

    def SetupFusedActivation(self):
        if self.tf_options == None:
            return
        try:
            activation_code = self.tf_options.FusedActivationFunction()
            self.activation = EnumStrMaps.ActivationFunctionType[activation_code]
        except AttributeError:
            # This operator does not support FusedActivationFunction
            pass
