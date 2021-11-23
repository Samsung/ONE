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
import torch.nn
import json
import zipfile


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
    torch.save(model.state_dict(), 'dict_model.pth')

    # create MAR with torchscript
    # create torchscript for MAR_torchscript test
    inp = torch.randn(1, 2, 3, 3)
    traced_model = torch.jit.trace(model, inp)
    torch.jit.save(traced_model, 'torchscript_model.pth')
    # TBD create manifest
    # example
    # MAR-INF/MANIFEST.json
    # {
    #     "createdOn": "16/11/2021 22:57:14",
    #     "runtime": "python",
    #     "model": {
    #         "modelName": "alexnet",
    #         "serializedFile": "alexnet.pt",
    #         "handler": "image_classifier",
    #         "modelVersion": "1.0"
    #     },
    #     "archiverVersion": "0.4.2"
    # }
    # TBD zip


    # create MAR with state_dict
    #TBD
    # MAR-INF/MANIFEST.json example
    # {
    #     "createdOn": "17/11/2021 13:09:26",
    #     "runtime": "python",
    #     "model": {
    #         "modelName": "strided_slice_eager",
    #         "serializedFile": "strided_slice_dict.pth",
    #         "handler": "image_classifier",
    #         "modelFile": "strided_slice.py",
    #         "modelVersion": "1.0"
    #     },
    #     "archiverVersion": "0.4.2"
    # }
