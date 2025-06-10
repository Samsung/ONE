#!/usr/bin/python3

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

import json
import os
from collections import OrderedDict
import sys
import argparse
import shutil


def verify(path):
    nnpackage_root_path = path

    # Check nnpackage_root existence
    if not os.path.isdir(nnpackage_root_path):
        print("Error: nnpackage_root {} does not exist.".format(nnpackage_root_path))
        sys.exit(-1)

    # Check MANIFEST existence
    manifest_path = os.path.join(nnpackage_root_path, "metadata", "MANIFEST")
    if not os.path.exists(manifest_path):
        print("Error: MANIFEST {} does not exist.".format(manifest_path))
        sys.exit(-1)

    # Check MANIFEST
    with open(manifest_path, "r") as f:
        try:
            json_dict = json.load(f, object_pairs_hook=OrderedDict)
            # Check models attributes
            for m in json_dict["models"]:
                model_path = os.path.join(nnpackage_root_path, m)
                if not os.path.exists(model_path):
                    print("Error: model {} does not exist.".format(model_path))
                    sys.exit(-1)
            print("nnpackage validation check passed.")
        except ValueError:
            print("MANIFEST is not valid JSON.")
        except KeyError:
            print("models attribute does not exist.")


def compress(path):
    nnpackage_name = os.path.basename(os.path.normpath(path))
    shutil.make_archive(nnpackage_name, 'zip', path)
    print("nnpackage compression is done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='the path to nnpackage')
    parser.add_argument('-v',
                        '--verify',
                        action='store_true',
                        help="verify nnpackage (default: false)")
    parser.add_argument('-c',
                        '--compress',
                        action='store_true',
                        help="compress nnpackage (default: false)")

    args = parser.parse_args()

    if args.verify:
        verify(args.path)

    if args.compress:
        compress(args.path)
