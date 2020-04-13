#!/usr/bin/python

# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

from config_saver import ConfigSaver


class ModelSaver(object):
    def __init__(self, model_name, op_parser):
        self.model_name = model_name
        self.op_parser = op_parser

    def SaveConfigInfo(self, prefix):
        print("Save model configuration file")
        for type_str, oper_list in self.op_parser.operators_per_type.items():
            if prefix:
                file_name = "{}_{}_{}.config".format(prefix, self.model_name, type_str)
            else:
                file_name = "{}_{}.config".format(self.model_name, type_str)
            print("{} file is generated".format(file_name))
            with open(file_name, 'wt') as f:
                f.write("# {}, Total count: {}\n\n".format(type_str, len(oper_list)))
            for operator in oper_list:
                ConfigSaver(file_name, operator).SaveInfo()
