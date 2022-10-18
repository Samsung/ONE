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

import os
import glob
import numpy as np

from pathlib import Path
from visqlib.Util import to_filename


class QErrorComputer:
    def __init__(self, fp32_dir, fq_dir):
        self._fp32_dir = fp32_dir
        self._fq_dir = fq_dir
        # Assumption: FM data are saved as follows
        #
        # fp32_dir/
        #   tensors.txt
        #   <DATA_INDEX>/
        #     <TENSOR_NAME>.npy
        #
        # fq_dir/
        #   tensors.txt
        #   <DATA_INDEX>/
        #     <TENSOR_NAME>.npy
        self._num_data = len(list(filter(os.path.isdir, glob.glob(fp32_dir + '/*'))))
        if self._num_data != len(list(filter(os.path.isdir, glob.glob(fq_dir + '/*')))):
            raise RuntimeError("Number of data mistmatches")

        self._filename_to_tensor = dict()
        with open(Path(fp32_dir) / 'tensors.txt') as f:
            tensors = set([line.rstrip() for line in f])
            for tensor in tensors:
                # Check if filename is unique
                # Fix name finding logic unless
                assert to_filename(tensor) not in self._filename_to_tensor
                self._filename_to_tensor[to_filename(tensor)] = tensor

    def run(self):
        '''Return qerror map (dict: tensor_name(string) -> qerror(float)).'''
        raise NotImplementedError  # Child must implement this


class MPEIRComputer(QErrorComputer):
    def __init__(self, fp32_dir, fq_dir):
        super().__init__(fp32_dir, fq_dir)

    def run(self):
        qerror_map = dict()
        for data_idx in range(self._num_data):
            fp32_results = glob.glob(self._fp32_dir + '/' + str(data_idx) + '/*.npy')
            for fp32_data_path in fp32_results:
                fp32_data = np.load(fp32_data_path)
                p = Path(fp32_data_path)
                fq_data_path = self._fq_dir + '/' + str(data_idx) + '/' + p.with_suffix(
                    '.npy').name
                fq_data = np.load(fq_data_path)

                diff = np.absolute(fp32_data - fq_data).reshape(-1)

                fp32_min = np.min(fp32_data.reshape(-1))
                fp32_max = np.max(fp32_data.reshape(-1))

                # Peak Error-to-Interval Ratio (PEIR)
                #
                # DESIGN NOTE: PEIR is an analogue of PSNR
                PEAK_ERROR = np.max(diff)
                INTERVAL = fp32_max - fp32_min

                # If INTERVAL is 0, PEIR becomes NaN.
                # To prevent this, relaxed PEIR with epsilon(10^(-6)) is used.
                rPEIR = PEAK_ERROR / (INTERVAL + 0.000001)

                filename = p.with_suffix('').name
                tensor_name = self._filename_to_tensor[filename]
                if tensor_name in qerror_map:
                    qerror_map[tensor_name] += rPEIR
                else:
                    qerror_map[tensor_name] = rPEIR

        for tensor_name, acc in qerror_map.items():
            qerror_map[tensor_name] = acc / self._num_data

        return qerror_map
