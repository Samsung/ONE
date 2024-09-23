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

import onelib.utils as oneutils
"""
[one hierarchy]
one
├── backends
├── bin
├── doc
├── include
├── lib
├── optimization
├── target
└── test

The list where `one-XXXX` finds its backends
- `bin` folder where `one-XXXX` exists
- `backends` folder

NOTE If there are backends of the same name in different places,
    the closer to the top in the list, the higher the priority.

[About TARGET and BACKEND]
  "Target" refers to an instance from the core of the system and
  "Backend" refers to an architecture. Say there is a NPU that has
  multiple cores. Its cores may have different global buffer 
  size, DSPM size and clock rate, etc, which are described in 
  each configuration file of "Target". Even though they
  are different target, they may follow same architecture, which means
  they have same "Backend".

[Path for TARGET configuration]
  - /usr/share/one/target/${TARGET}.ini

[Path for BACKEND tools]
  - /usr/share/one/backends/${BACKEND}
"""


def get_list(cmdname):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    backend_set = set()

    # bin folder
    files = [f for f in glob.glob(dir_path + '/../' + cmdname)]
    # backends folder
    files += [
        f for f in glob.glob(dir_path + '/../../backends/**/' + cmdname, recursive=True)
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


def get_value_from_target_conf(target: str, key: str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    target_conf_path = dir_path + f'/../../target/{target}.ini'
    if not os.path.isfile(target_conf_path):
        raise FileNotFoundError(f"Not found given target configuration: {target}")

    # target config doesn't have section.
    # but, configparser needs configs to have one or more sections.
    DUMMY_SECTION = 'dummy_section'
    with open(target_conf_path, 'r') as f:
        config_str = f'[{DUMMY_SECTION}]\n' + f.read()
    parser = oneutils.get_config_parser()
    parser.read_string(config_str)
    assert parser.has_section(DUMMY_SECTION)

    # Check if target file is valid
    TARGET_KEY = 'TARGET'
    assert TARGET_KEY in parser[DUMMY_SECTION]
    if target != parser[DUMMY_SECTION][TARGET_KEY]:
        raise RuntimeError("Invalid target file.")

    if key in parser[DUMMY_SECTION]:
        return parser[DUMMY_SECTION][key]

    raise RuntimeError(f"Not found '{key}' key in target configuration.")


def search_driver(driver):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # CASE 1: one/bin/{driver} is found
    driver_path = dir_path + '/../' + driver
    if os.path.isfile(driver_path) and os.access(driver_path, os.X_OK):
        return driver_path

    # CASE 2: one/backends/**/bin/{driver} is found
    for driver_path in glob.glob(dir_path + '/../../backends/**/bin/' + driver,
                                 recursive=True):
        if os.path.isfile(driver_path) and os.access(driver_path, os.X_OK):
            return driver_path

    # CASE 3: {driver} is found in nowhere
    return None
