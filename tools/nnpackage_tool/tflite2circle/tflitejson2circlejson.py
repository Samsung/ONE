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
import sys
from collections import OrderedDict


def usage():
    script = os.path.basename(os.path.basename(__file__))
    print("Usage: {} path_to_tflite_in_json".format(script))
    sys.exit(-1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()

    json_path = sys.argv[1]
    with open(json_path, "r") as f:
        try:
            json_dict = json.load(f, object_pairs_hook=OrderedDict)
            json_dict["version"] = 0
            print(json.dumps(json_dict, indent=2))
        except KeyError:
            print("subgraphs attribute does not exist.")
            sys.exit(-2)
