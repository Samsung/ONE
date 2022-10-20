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

# Script that dumps dequantized FM
# NOTE This script runs on dalgona

import numpy as np

from pathlib import Path
from Util import to_filename

# Fake-quantized Op has the postfix of fq_postfix
# TODO Remove coupling with fake quantization codes
fq_postfix = '_FQ_Quantize_FQ_Dequantize'


# Return the original name before fake quantization
# Return None if name is not from fake quantization (Dequantize Op in original model)
# TODO Handle the case when the original node's name contains fq_postfix
def _name_before_fq(name):
    if not name.endswith(fq_postfix):
        return None

    return name[0:name.find(fq_postfix)]


# Dump fake-quantized model's intermediate FM data according to tensors.txt
#
# Before
# self._dir/
#  tensors.txt
#
# After
# self._dir/
#  tensors.txt
#  <TENSOR_NAME>.npy
# NOTE TENSOR_NAME is transformed by to_filename
class DumpFakeQuantFM:
    def StartAnalysis(self, args):
        self._dir = Path(args)
        self._num_data = 0
        with open(self._dir / 'tensors.txt') as f:
            self._target_tensors = set([line.rstrip() for line in f])

    def EndNetworkExecution(self, outputs: list):
        self._num_data += 1

    # TODO Use DequantizePost when dalgona supports it
    def DefaultOpPost(self, name, opcode, inputs, output):
        if opcode == 'Dequantize':
            orig_name = _name_before_fq(name)
            if orig_name in self._target_tensors:
                data_path = self._dir / str(self._num_data)
                data_path.mkdir(parents=False, exist_ok=True)
                np.save(str(data_path / to_filename(orig_name)), output['data'])
