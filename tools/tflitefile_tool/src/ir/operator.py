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


class Operator(object):
    def __init__(self):
        self._index = -1
        self._inputs = []
        self._outputs = []
        self._op_name = ""
        self._actviation = ""
        self._options = ""

    '''index'''

    @property
    def index(self):
        '''operator's int type index'''
        return self._index

    @index.setter
    def index(self, value):
        if not isinstance(value, int):
            raise TypeError("must be set to an integer")
        self._index = value

    '''inputs'''

    @property
    def inputs(self):
        '''Operators's input tensors as a list which consists of Tensors'''
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if not isinstance(value, list):
            raise TypeError("must be set to a list")
        self._inputs = value

    '''outputs'''

    @property
    def outputs(self):
        '''Operators's output tensors as a list which consists of Tensors'''
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        if not isinstance(value, list):
            raise TypeError("must be set to a list")
        self._outputs = value

    '''op_name'''

    @property
    def op_name(self):
        '''Operator's name str'''
        return self._op_name

    @op_name.setter
    def op_name(self, value):
        if not isinstance(value, str):
            raise TypeError("must be set to a str")
        self._op_name = value

    '''actviation'''

    @property
    def actviation(self):
        '''Operator's actviation str'''
        return self._actviation

    @actviation.setter
    def actviation(self, value):
        if not isinstance(value, str):
            raise TypeError("must be set to a str")
        self._actviation = value

    '''options'''

    @property
    def options(self):
        '''Operator's options str'''
        return self._options

    @options.setter
    def options(self, value):
        if not isinstance(value, str):
            raise TypeError("must be set to a str")
        self._options = value
