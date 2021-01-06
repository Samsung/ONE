#!/usr/bin/env python3

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

from pathlib import Path


class OpListParser():
    """
    Read op_list.txt to create supported operation list for each backend

    TODO : Reads supported tensor type for each operation (FP32 or INT8)
    """
    def __init__(self):
        self.file_name = "op_list.txt"
        self.op_list_file = Path(__file__).parent / self.file_name

    def parse(self):
        backend_op_list = {}
        with open(self.op_list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                backend, _, op_list_str = line.partition(':')
                op_list = op_list_str.split(',')
                backend_op_list[backend] = op_list
        return backend_op_list
