# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

# Script that dumps FM of FP32 model
# NOTE This script runs on dalgona

import numpy as np
import json

from pathlib import Path


# Dump FP32 model's intermediate FM data and their names
#
# Before
# self._dir/
#
# After
# self._dir/
#  tensors.json
#  <TENSOR_ID>.npy
# NOTE tensors.json has a dictionary {TENSOR_NAME -> TENSOR_ID}
class DumpFP32FM:
    def StartAnalysis(self, args):
        self._dir = Path(args)
        self._num_data = 0
        # Dict {tensor_name -> tid}
        self._tname_to_tid = dict()
        self._tensor_count = 0

    def EndNetworkExecution(self, outputs):
        self._num_data += 1

    def DefaultOpPost(self, name, opcode, inputs, outputs):
        # Save intermediate FM into <tid>.npy
        data_path = self._dir / str(self._num_data)
        data_path.mkdir(parents=False, exist_ok=True)
        for output in outputs:
            name = output['name']
            data = output['data']
            if name in self._tname_to_tid:
                tid = self._tname_to_tid[name]
            else:
                tid = self._tensor_count
                self._tname_to_tid[name] = tid
                self._tensor_count += 1

            np.save(str(data_path / str(tid)), data)

    def EndAnalysis(self):
        # Save tensor name : tensor id pairs
        with open(self._dir / 'tensors.json', 'w') as f:
            json.dump(self._tname_to_tid, f, indent=2)
