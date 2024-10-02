#!/usr/bin/env python3

# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

import onnx
import os
import sys


def _help_exit(cmd_name):
    print('Produce shape-infered ONNX file')
    print('Usage: {0} [onnx_in_path] [onnx_out_path]'.format(cmd_name))
    print('')
    exit()


def main():
    if len(sys.argv) < 3:
        _help_exit(os.path.basename(sys.argv[0]))

    onnx.checker.check_model(sys.argv[1])
    onnx.shape_inference.infer_shapes_path(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
