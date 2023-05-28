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

from tensor_wrapping import Tensor

SYMBOLS = ['B', 'K', 'M', 'G', 'T']


def ConvertBytesToHuman(n):
    n = int(n)
    if n < 0:
        return 0

    format_str = "%(val)3.1f%(symb)s"
    prefix = {}
    for i, s in enumerate(SYMBOLS[1:]):
        prefix[s] = 1 << (i + 1) * 10

    for symbol in reversed(SYMBOLS[1:]):
        if n >= prefix[symbol]:
            v = float(n) / prefix[symbol]
            return format_str % dict(symb=symbol, val=v)

    return format_str % dict(symb=SYMBOLS[0], val=n)


class TensorPrinter(object):
    def __init__(self, verbose, tensor):
        self.verbose = verbose
        self.tensor = tensor

    def PrintInfo(self, depth_str=""):
        if (self.verbose < 1):
            pass

        print_str = ""
        if self.tensor.tensor_idx < 0:
            print_str = "Tensor {0:4}".format(self.tensor.tensor_idx)
        else:
            buffer_idx = self.tensor.tf_tensor.Buffer()
            isEmpty = "Filled"
            if (self.tensor.tf_buffer.DataLength() == 0):
                isEmpty = " Empty"
            shape_str = self.GetShapeString()
            type_name = self.tensor.type_name

            shape_name = ""
            if self.tensor.tf_tensor.Name() != 0:
                shape_name = self.tensor.tf_tensor.Name()

            memory_size = ConvertBytesToHuman(self.tensor.memory_size)

            print_str = "Tensor {0:4} : buffer {1:4} | {2} | {3:7} | Memory {4:6} | Shape {5} ({6})".format(
                self.tensor.tensor_idx, buffer_idx, isEmpty, type_name, memory_size,
                shape_str, shape_name)
        print(depth_str + print_str)

    def GetShapeString(self):
        if self.tensor.tf_tensor.ShapeLength() == 0:
            return "Scalar"
        return_string = "["
        for shape_idx in range(self.tensor.tf_tensor.ShapeLength()):
            if (shape_idx != 0):
                return_string += ", "
            # when shape signature is -1, that means unknown dim
            if self.tensor.tf_tensor.ShapeSignature(shape_idx) != -1:
                return_string += str(self.tensor.tf_tensor.Shape(shape_idx))
            else:
                return_string += "-1"
        return_string += "]"
        return return_string
