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

import tflite.BuiltinOperator
import tflite.ActivationFunctionType
import tflite.BuiltinOptions


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
