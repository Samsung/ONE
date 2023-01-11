#!/usr/bin/env python

# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

import utils as oneutils


class Command:
    def __init__(self, driver, args, log_file):
        self.cmd = [driver]
        self.driver = driver
        self.args = args
        self.log_file = log_file

    # Add option if attrs are valid
    # Option values are collected from self.args
    def add_option_with_valid_args(self, option, attrs):
        for attr in attrs:
            if not oneutils._is_valid_attr(self.args, attr):
                return self
        self.cmd.append(option)
        for attr in attrs:
            self.cmd.append(getattr(self.args, attr))
        return self

    # Add option and values without any condition
    def add_option_with_values(self, option, values):
        self.cmd.append(option)
        for value in values:
            self.cmd.append(value)
        return self

    # Add option with no argument (ex: --verbose) if attr is valid
    def add_noarg_option_if_valid_arg(self, option, attr):
        if oneutils._is_valid_attr(self.args, attr):
            self.cmd.append(option)
        return self

    # Run cmd and save logs
    def run(self):
        self.log_file.write((' '.join(self.cmd) + '\n').encode())
        oneutils._run(self.cmd, err_prefix=self.driver, logfile=self.log_file)
