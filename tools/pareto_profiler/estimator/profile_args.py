#! /usr/bin/python

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

import argparse


class ProfileArgs(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ProfileArgs, self).__init__(args, kwargs)
        self.add_argument(
            'model', type=str, default=None, help='nnpackage name with path')
        self.add_argument('run_folder', type=str, help="path to nnpackage_run executable")
        self.add_argument(
            '--mode',
            type=str.lower,
            choices=["index", "name"],
            default="name",
            help='Profile by operation index or name')
        self.add_argument('--backends', type=int, default=2, help='Number of backends')
        self.add_argument(
            '--dumpfile',
            type=str.lower,
            default="/tmp/final_result.json",
            help='JSON Dumpfile name with path')
