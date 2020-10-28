#!/usr/bin/env python

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

# PyTorch Example manager

import torch
import importlib
import argparse

parser = argparse.ArgumentParser(description='Process PyTorch python examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

for example in args.examples:
    module = importlib.import_module("examples." + example)
    torch.save(module._model_, example + ".pth")
    print("Generate '" + example + ".pth' - Done")

    torch.onnx.export(module._model_, module._dummy_, example + ".onnx")
    print("Generate '" + example + ".onnx' - Done")
