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

import glob
import ntpath
import os
"""
[one hierarchy]
one
├── backends
├── bin
├── doc
├── include
├── lib
├── optimization
└── test

The list where `one-XXXX` finds its backends
- `bin` folder where `one-XXXX` exists
- `backends` folder

NOTE If there are backends of the same name in different places,
    the closer to the top in the list, the higher the priority.
"""


def get_list(cmdname):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    backend_set = set()

    # bin folder
    files = [f for f in glob.glob(dir_path + '/../*-' + cmdname)]
    # backends folder
    files += [
        f
        for f in glob.glob(dir_path + '/../../backends/**/*-' + cmdname, recursive=True)
    ]
    # TODO find backends in `$PATH`

    backends_list = []
    for cand in files:
        base = ntpath.basename(cand)
        if (not base in backend_set) and os.path.isfile(cand) and os.access(
                cand, os.X_OK):
            backend_set.add(base)
            backends_list.append(cand)

    return backends_list


def search_driver(driver):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # CASE 1: one/bin/{driver} is found
    driver_path = dir_path + '/../' + driver
    if os.path.isfile(driver_path) and os.access(driver_path, os.X_OK):
        return driver_path

    # CASE 2: one/backends/**/bin/{driver} is found
    for driver_path in glob.glob(
            dir_path + '/../../backends/**/bin/' + driver, recursive=True):
        if os.path.isfile(driver_path) and os.access(driver_path, os.X_OK):
            return driver_path

    # CASE 3: {driver} is found in nowhere
    return None
