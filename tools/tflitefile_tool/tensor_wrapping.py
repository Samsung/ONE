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

import tflite.Tensor
import tflite.TensorType

TensorTypeList = {}


def SetTensorTypeStr():
    tensorTypeObj = tflite.TensorType.TensorType()

    for fieldName in dir(tensorTypeObj):
        if (not fieldName.startswith('_')):
            fieldValue = getattr(tensorTypeObj, fieldName)
            if (isinstance(fieldValue, (int))):
                TensorTypeList[fieldValue] = fieldName


TYPES = {
    'BOOL': 1,
    'COMPLEX64': 8,
    'FLOAT16': 2,
    'FLOAT32': 4,
    'INT16': 2,
    'INT32': 4,
    'INT64': 8,
    'UINT8': 1
}


def GetTypeSize(type_name):
    try:
        return TYPES[type_name]

    except KeyError as error:
        return 0


class Tensor(object):
    def __init__(self, tensor_idx, tf_tensor, tf_buffer):
        self.tensor_idx = tensor_idx
        self.tf_tensor = tf_tensor
        self.tf_buffer = tf_buffer

        # optional input
        if (self.tf_tensor != None):
            self.type_name = TensorTypeList[self.tf_tensor.Type()]
        else:
            self.type_name = None

        self.memory_size = self.GetMemorySize()

    def GetMemorySize(self):
        type_size = GetTypeSize(self.type_name)
        if type_size == 0:
            return 0

        # memory size in bytes
        size = int(type_size)
        shape_length = self.tf_tensor.ShapeLength()
        if shape_length == 0:
            return size

        for shape_idx in range(shape_length):
            shape_size = int(self.tf_tensor.Shape(shape_idx))
            size *= shape_size

        return size
