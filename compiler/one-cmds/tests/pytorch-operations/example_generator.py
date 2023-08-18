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

# PyTorch Example manager

import torch
import importlib.machinery
import importlib.util
import argparse
import os

from pathlib import Path

print("PyTorch version=", torch.__version__)

parser = argparse.ArgumentParser(description='Process PyTorch python examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

output_folder = "./"

Path(output_folder).mkdir(parents=True, exist_ok=True)


class JitWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        if len(args) == 1:
            return self.model.forward(args[0])
        else:
            return self.model.forward(args)


for example in args.examples:
    print("Generate '" + example + ".pth'", end='')
    # load example code
    # replace - with _ in name, otherwise pytorch generates invalid torchscript
    module_name = "examples." + example.replace('-', '_')
    module_loader = importlib.machinery.SourceFileLoader(
        module_name, os.path.join("examples", example, "__init__.py"))
    module_spec = importlib.util.spec_from_loader(module_name, module_loader)
    module = importlib.util.module_from_spec(module_spec)
    module_loader.exec_module(module)

    jittable_model = JitWrapper(module._model_)

    traced_model = torch.jit.trace(jittable_model, module._dummy_)
    # save .pth
    torch.jit.save(traced_model, output_folder + example + ".pth")

    input_shapes = ""
    input_types = ""

    input_samples = module._dummy_
    if isinstance(input_samples, torch.Tensor):
        input_samples = [input_samples]
    for inp_idx in range(len(input_samples)):
        input_data = input_samples[inp_idx]

        shape = input_data.shape
        for dim in range(len(shape)):
            input_shapes += str(shape[dim])
            if dim != len(shape) - 1:
                input_shapes += ","

        if input_data.dtype == torch.bool:
            input_types += "bool"
        elif input_data.dtype == torch.uint8:
            input_types += "uint8"
        elif input_data.dtype == torch.int8:
            input_types += "int8"
        elif input_data.dtype == torch.int16:
            input_types += "int16"
        elif input_data.dtype == torch.int32:
            input_types += "int32"
        elif input_data.dtype == torch.int64:
            input_types += "int16"
        elif input_data.dtype == torch.float16:
            input_types += "float32"
        elif input_data.dtype == torch.float32:
            input_types += "float32"
        elif input_data.dtype == torch.float64:
            input_types += "float64"
        elif input_data.dtype == torch.complex64:
            input_types += "complex64"
        elif input_data.dtype == torch.complex128:
            input_types += "complex128"
        else:
            raise ValueError('unsupported dtype')

        if inp_idx != len(input_samples) - 1:
            input_shapes += ":"
            input_types += ","

    with open(example + ".spec", "w") as spec_file:
        print(input_shapes, file=spec_file)
        print(input_types, file=spec_file)

    print(" - Done")
