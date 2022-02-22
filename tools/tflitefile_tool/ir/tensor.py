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
"""
NOTE
- This class expresses a wrapping class for a native class.
- Just use this class as an interface.
"""


class Tensor(object):
    def __init__(self):
        self._index = -1
        self._tensor_name = ""
        self._buffer = None
        self._buffer_index = -1
        self._type_name = ""
        self._shape = []
        self._memory_size = -1

    '''index'''

    @property
    def index(self):
        '''Tensor's int type index'''
        return self._index

    @index.setter
    def index(self, value):
        if not isinstance(value, int):
            raise TypeError("must be set to an integer")
        self._index = value

    '''tensor_name'''

    @property
    def tensor_name(self):
        '''Tensor's name str'''
        return self._tensor_name

    @tensor_name.setter
    def tensor_name(self, value):
        if not isinstance(value, str):
            raise TypeError("must be set to a str")
        self._tensor_name = value

    '''buffer'''

    @property
    def buffer(self):
        '''Tensor's buffer as a numpy instance type'''
        return self._buffer

    @buffer.setter
    def buffer(self, value):
        self._buffer = value

    '''buffer_index'''

    @property
    def buffer_index(self):
        '''Tensor's int type buffer index'''
        return self._buffer_index

    @buffer_index.setter
    def buffer_index(self, value):
        if not isinstance(value, int):
            raise TypeError("must be set to an integer")
        self._buffer_index = value

    '''type_name'''

    @property
    def type_name(self):
        '''Tensor's type name str'''
        return self._type_name

    @type_name.setter
    def type_name(self, value):
        if not isinstance(value, str):
            raise TypeError("must be set to a str")
        self._type_name = value

    '''shape'''

    @property
    def shape(self):
        '''Tensor's shape as a list'''
        return self._shape

    @shape.setter
    def shape(self, value):
        if not isinstance(value, list):
            raise TypeError("must be set to a list")
        self._shape = value

    '''memory_size'''

    @property
    def memory_size(self):
        '''Tensor's memory size as int type'''
        return self._memory_size

    @memory_size.setter
    def memory_size(self, value):
        if not isinstance(value, int):
            raise TypeError("must be set to an integer")
        self._memory_size = value
