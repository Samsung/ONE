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

# Script that dumps FM of fp32 model
# NOTE This script runs on dalgona

import numpy as np

from pathlib import Path
from Util import to_filename


# Dump fp32 model's intermediate FM data and their names
#
# Before
# self._dir/
#
# After
# self._dir/
#  tensors.txt
#  <TENSOR_NAME>.npy
# NOTE TENSOR_NAME is transformed by to_filename
class DumpFP32FM(object):
    def StartAnalysis(self, args):
        self._dir = Path(args)
        self._num_data = 0
        self._tensor_names = set()

    def EndNetworkExecution(self, outputs):
        self._num_data += 1

    def DefaultOpPost(self, name, opcode, inputs, output):
        # Save intermediate FM into tensor_name.npy
        data_path = self._dir / str(self._num_data)
        data_path.mkdir(parents=False, exist_ok=True)
        np.save(str(data_path / to_filename(name)), output['data'])
        self._tensor_names.add(name)

    def EndAnalysis(self):
        # Save tensor names line by line
        with open(self._dir / 'tensors.txt', 'w') as f:
            for name in self._tensor_names:
                f.write("%s\n" % name)
