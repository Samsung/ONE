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

from .string_builder import StringBuilder


# TODO: Extract to a single Printer class like Printer.print(option)
class OptionPrinter(object):
    def __init__(self, verbose, op_name, options):
        self.verbose = verbose
        self.op_name = op_name
        self.options = options

    def PrintInfo(self, tab=""):
        info = StringBuilder(self.verbose).Option(self.op_name, self.options, tab)
        if info is not None:
            print(info)
