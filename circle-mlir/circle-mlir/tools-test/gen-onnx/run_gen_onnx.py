# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import torch
import importlib
import sys

from pathlib import Path


def generate_onnx(models_root, model_name, onnx_file):
    sys.path.append(models_root)
    module = importlib.import_module(model_name)

    # default: refer https://github.com/pytorch/pytorch/blob/master/torch/onnx/utils.py
    # and https://github.com/pytorch/pytorch/blob/master/torch/onnx/_constants.py
    # and https://github.com/pytorch/pytorch/blob/master/tools/onnx/update_default_opset_version.py
    opset_version = 14
    if hasattr(module._model_, 'onnx_opset_version'):
        opset_version = module._model_.onnx_opset_version()

    m_keys = module.__dict__.keys()

    if '_io_names_' in m_keys and '_dynamic_axes_' in m_keys:
        # refer https://github.com/onnx/onnx/issues/654#issuecomment-521233285
        # purpose of this is to set dynamic shape for inputs or inputs
        # magic(?) is to set input/output names, and then set dyanmic shape by name/dim
        # example) set output dim(0) as unknown
        #    _io_names_ = [['input'], ['output']]
        #    _dynamic_axes_ = {'output': {0: '?'}}
        torch.onnx.export(module._model_,
                          module._inputs_,
                          onnx_file,
                          input_names=module._io_names_[0],
                          output_names=module._io_names_[1],
                          dynamic_axes=module._dynamic_axes_,
                          opset_version=opset_version)
    else:
        torch.onnx.export(module._model_,
                          module._inputs_,
                          onnx_file,
                          opset_version=opset_version)

    if hasattr(module._model_, 'post_process'):
        module._model_.post_process(onnx_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        thispath = Path(sys.argv[0])
        sys.exit("Usage: " + thispath.name + " [models_root] [model_name] [onnx_file]")

    generate_onnx(sys.argv[1], sys.argv[2], sys.argv[3])
