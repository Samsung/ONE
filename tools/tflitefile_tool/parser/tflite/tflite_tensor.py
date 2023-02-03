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

import numpy as np
import tflite.Tensor
import tflite.TensorType
from ir.tensor import Tensor

TensorTypeList = {}


def SetTensorTypeStr():
    tensorTypeObj = tflite.TensorType.TensorType()

    for fieldName in dir(tensorTypeObj):
        if (not fieldName.startswith('_')):
            fieldValue = getattr(tensorTypeObj, fieldName)
            if (isinstance(fieldValue, (int))):
                TensorTypeList[fieldValue] = fieldName


TYPES_SIZE = {
    'BOOL': 1,
    'COMPLEX64': 8,
    'FLOAT16': 2,
    'FLOAT32': 4,
    'INT16': 2,
    'INT32': 4,
    'INT64': 8,
    'UINT8': 1,
    'NONE': 0,
}


def GetTypeSize(type_name):
    try:
        return TYPES_SIZE[type_name]

    except KeyError as error:
        return 0


TYPE_TO_NPTYPE = {
    'BOOL': np.bool_,
    'COMPLEX64': np.cdouble,
    'FLOAT16': np.float16,
    'FLOAT32': np.float32,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
}


def ConvertProperNPArrayType(np_arr, np_shape, type_name):
    try:
        return np_arr.view(TYPE_TO_NPTYPE[type_name]).reshape(np_shape)
    except KeyError as error:
        return np_arr.view().reshape(np_shape)


class TFLiteTensor(Tensor):
    def __init__(self, tensor_idx, tf_tensor, tf_buffer):
        super(TFLiteTensor, self).__init__()
        self.tf_tensor = tf_tensor
        self.tf_buffer = tf_buffer

        self.index = int(tensor_idx)
        self.tensor = tf_tensor

        # optional input
        if self.index == -1:
            self.type_name = "NONE"
        # general input
        else:
            assert tf_tensor is not None
            assert tf_buffer is not None
            self.tensor_name = str(tf_tensor.Name())
            self.type_name = TensorTypeList[tf_tensor.Type()]
            self.buffer_index = tf_tensor.Buffer()
            if (tf_buffer.DataLength() > 0):
                self.buffer = ConvertProperNPArrayType(tf_buffer.DataAsNumpy(),
                                                       tf_tensor.ShapeAsNumpy(),
                                                       self.type_name)

            # shape: Empty list([]) will mean Scalar
            for shape_idx in range(tf_tensor.ShapeLength()):
                # when shape signature is -1, that means unknown dim
                if tf_tensor.ShapeSignature(shape_idx) != -1:
                    self.shape.append(int(tf_tensor.Shape(shape_idx)))
                else:
                    self.shape.append(-1)

        self.memory_size = self.GetMemorySize()

    def GetMemorySize(self):
        type_size = GetTypeSize(self.type_name)
        if type_size == 0:
            return 0

        # memory size in bytes
        size = int(type_size)
        shape_length = len(self.shape)
        if shape_length == 0:
            return size

        for shape_idx in range(shape_length):
            shape_size = int(self.shape[shape_idx])
            size *= shape_size

        return size
