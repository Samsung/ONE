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

# PyTorch aux tests generator

import torch
import torch.nn as nn
import json
import zipfile
import os


# model
class net_abs(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, input):
        return torch.abs(input)


if __name__ == '__main__':
    model = net_abs()
    # save "entire" model for entire_model.test
    torch.save(model, 'entire_model.pth')

    # save state_dict file for state_dict.test
    state_dict_path = 'dict_model.pth'
    torch.save(model.state_dict(), state_dict_path)

    # create files for mar_torchscript.test
    torchscript_path = 'torchscript_model.pth'
    inp = torch.randn(1, 2, 3, 3)
    traced_model = torch.jit.trace(model, inp)
    torch.jit.save(traced_model, torchscript_path)
    # create manifest
    manifest = {}
    manifest['createdOn'] ='11/11/1111 11:11:11'
    manifest['runtime'] = 'python'
    manifest['model'] = {}
    manifest['model']['modelName'] = 'torchscript_model',
    manifest['model']['serializedFile'] = torchscript_path
    manifest['model']['handler'] = 'image_classifier'
    manifest['model']['modelVersion'] = '1.0'
    manifest['archiverVersion'] = '0.4.2'

    with zipfile.ZipFile('mar_torchscript.mar', 'w') as mar_file:
        with mar_file.open('MAR-INF/MANIFEST.json', 'w') as manifest_file:
            manifest_file.write(json.dumps(manifest).encode())
        mar_file.write(torchscript_path)

    # create files for mar_state_dict.test
    model_file_path = os.path.basename(__file__)
    # create manifest
    manifest = {}
    manifest['createdOn'] ='11/11/1111 11:11:11'
    manifest['runtime'] = 'python'
    manifest['model'] = {}
    manifest['model']['modelName'] = 'state_dict_model',
    manifest['model']['serializedFile'] = state_dict_path
    manifest['model']['handler'] = 'image_classifier'
    manifest['model']['modelFile'] = model_file_path
    manifest['model']['modelVersion'] = '1.0'
    manifest['archiverVersion'] = '0.4.2'

    with zipfile.ZipFile('mar_state_dict.mar', 'w') as mar_file:
        with mar_file.open('MAR-INF/MANIFEST.json', 'w') as manifest_file:
            manifest_file.write(json.dumps(manifest).encode())
        mar_file.write(state_dict_path)
        mar_file.write(model_file_path)
