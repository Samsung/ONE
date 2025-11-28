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

from collections.abc import MutableMapping
'''optype -> Operator Index List'''


class OpTypesMap(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        k = self._keytransform(key)
        if not k in self.store.keys():
            self.store[k] = []
        self.store[k].append(value)

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        if not isinstance(key, str):
            raise TypeError("must be set to a str")
        return key


"""
NOTE
- This class expresses a wrapping class for a native class.
- Just use this class as an interface.
"""


class Subgraph(object):
    def __init__(self):
        self._index = -1
        self._inputs = []
        self._outputs = []
        self._subg_name = ""
        self._model_name = ""
        self._tensors_map = {}
        self._operators_map = {}
        self._optypes_map = OpTypesMap()

    '''index'''

    @property
    def index(self):
        '''Subgraph's int type index'''
        return self._index

    @index.setter
    def index(self, value):
        if not isinstance(value, int):
            raise TypeError("must be set to an integer")
        self._index = value

    '''inputs'''

    @property
    def inputs(self):
        '''Subgraph's input tensors as a list which consists of Tensors'''
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if not isinstance(value, list):
            raise TypeError("must be set to a list")
        self._inputs = value

    '''outputs'''

    @property
    def outputs(self):
        '''Subgraph's output tensors as a list which consists of Tensors'''
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        if not isinstance(value, list):
            raise TypeError("must be set to a list")
        self._outputs = value

    '''subg_name'''

    @property
    def subg_name(self):
        '''Subgraph's name str'''
        return self._subg_name

    @subg_name.setter
    def subg_name(self, value):
        if not isinstance(value, str):
            raise TypeError("must be set to a str")
        self._subg_name = value

    '''model_name'''

    @property
    def model_name(self):
        '''Model name str'''
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if not isinstance(value, str):
            raise TypeError("must be set to a str")
        self._model_name = value

    '''tensors_map'''

    @property
    def tensors_map(self):
        '''Subgraph's all tensors(key:index, value:Tensor)'''
        return self._tensors_map

    @tensors_map.setter
    def tensors_map(self, value):
        if not isinstance(value, dict):
            raise TypeError("must be set to a dict")
        self._tensors_map = value

    '''operators_map'''

    @property
    def operators_map(self):
        '''Subgraph's operators(key:index, value:Operator)'''
        return self._operators_map

    @operators_map.setter
    def operators_map(self, value):
        if not isinstance(value, dict):
            raise TypeError("must be set to a dict")
        self._operators_map = value

    '''optypes_map'''

    @property
    def optypes_map(self):
        '''Subgraph's operators per type(key:optype, value:[op_indice])'''
        return self._optypes_map

    @optypes_map.setter
    def optypes_map(self, value):
        if not isinstance(value, OpTypesMap):
            raise TypeError("must be set to a OpTypesMap")
        self._optypes_map = value
