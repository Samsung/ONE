#!/usr/bin/env python3

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

import json, logging
from distutils.dir_util import copy_tree
from pathlib import Path


class NnpkgHelper:
    """
    Helper class for nnpackage
    """

    def __init__(self):
        self.config_name = 'config.cfg'

    def copy(self, src, dst):
        copy_tree(str(src), str(dst))

    def add_config(self, src, configs):
        manifest_path = Path(src).resolve() / 'metadata' / 'MANIFEST'
        config_path = Path(src).resolve() / 'metadata' / self.config_name

        try:
            # Read MANIFEST file
            with open(manifest_path, 'r') as manifest_file:
                data = json.load(manifest_file)

            # Add configs to MANIFEST file
            with open(manifest_path, 'w') as manifest_file:
                data['configs'] = [self.config_name]
                json.dump(data, manifest_file, indent=2)

            # Write config.cfg file
            with open(config_path, 'w') as config_file:
                config_file.write('\n'.join(configs))

            logging.info(f"Scheduled nnpackage is saved at {src}")

        except IOError as e:
            logging.warn(e)
        except:
            logging.warn("Error")
