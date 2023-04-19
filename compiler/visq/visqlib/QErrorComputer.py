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
import json

from pathlib import Path
from visqlib.Util import to_filename
from collections import defaultdict


class QErrorComputer:
    def __init__(self, fp32_dir, fq_dir):
        self._fp32_dir = fp32_dir
        self._fq_dir = fq_dir
        self.qerror_map = defaultdict(float)
        self._num_processed_data = 0

    def collect_data_path(self, fp32_dir, fq_dir):
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

        self._num_processed_data += self._num_data

        self._filename_to_tensor = dict()
        with open(Path(fp32_dir) / 'tensors.txt') as f:
            tensors = set([line.rstrip() for line in f])
            for tensor in tensors:
                # Check if filename is unique
                # Fix name finding logic unless
                assert to_filename(tensor) not in self._filename_to_tensor
                self._filename_to_tensor[to_filename(tensor)] = tensor

        # Save paths to fp32 data and fq data for each tensor
        # dict
        # {
        #   <tensor_name>: (fp32_path, fq_path),
        #   <tensor_name>: (fp32_path, fq_path),
        #   ...
        # }
        data_paths = dict()
        for data_idx in range(self._num_data):
            fp32_results = glob.glob(fp32_dir + '/' + str(data_idx) + '/*.npy')
            for fp32_data_path in fp32_results:
                fp32_path = Path(fp32_data_path)
                fq_data_path = fq_dir + '/' + str(data_idx) + '/' + fp32_path.with_suffix(
                    '.npy').name
                fq_path = Path(fq_data_path)
                filename = fp32_path.stem
                tensor_name = self._filename_to_tensor[filename]

                # Only save the tensors which have both fp32 data and fq data
                if fq_path.is_file() and fp32_path.is_file():
                    if tensor_name in data_paths:
                        data_paths[tensor_name].append((fp32_data_path, fq_data_path))
                    else:
                        data_paths[tensor_name] = [(fp32_data_path, fq_data_path)]

        return data_paths

    def run(self):
        '''Return qerror map (dict: tensor_name(string) -> qerror(float)).'''
        raise NotImplementedError  # Child must implement this


class MPEIRComputer(QErrorComputer):
    def __init__(self, fp32_dir, fq_dir):
        super().__init__(fp32_dir, fq_dir)

    # Incrementally compute Qerror while traversing all data in fp32_dir and fq_dir
    def advance_on(self, fp32_dir, fq_dir):
        data_paths = self.collect_data_path(fp32_dir, fq_dir)
        for tensor_name, data_path in data_paths.items():
            for (fp32_data_path, fq_data_path) in data_path:
                fp32_data = np.load(fp32_data_path)
                fq_data = np.load(fq_data_path)

                diff = np.absolute(fp32_data - fq_data).reshape(-1)

                fp32_min = np.min(fp32_data.reshape(-1))
                fp32_max = np.max(fp32_data.reshape(-1))

                # Peak Error-to-Interval Ratio (PEIR)
                # NOTE: PEIR is an analogue of PSNR (Peak Signal to Noise Ratio)
                PEAK_ERROR = np.max(diff)
                INTERVAL = fp32_max - fp32_min

                # If INTERVAL is 0, PEIR becomes NaN.
                # To prevent this, relaxed PEIR with epsilon(10^(-6)) is used.
                rPEIR = PEAK_ERROR / (INTERVAL + 0.000001)

                self.qerror_map[tensor_name] += rPEIR

    # Return
    # qerror_map (dict: tensor_name(string) -> qerror(float))
    # qerror_min (float)
    # qerror_max (float)
    def get_final_result(self):
        qerror_map = dict()
        for tensor_name, acc in self.qerror_map.items():
            qerror_map[tensor_name] = acc / self._num_processed_data

        # Fixed qerror_min (0), qerror_max (1)
        return qerror_map, 0.0, 1.0

    def run(self):
        self.advance_on(self._fp32_dir, self._fq_dir)
        return self.get_final_result()


class MSEComputer(QErrorComputer):
    def __init__(self, fp32_dir, fq_dir):
        super().__init__(fp32_dir, fq_dir)
        self.qerror_min = float('inf')
        self.qerror_max = -self.qerror_min

    # Incrementally compute Qerror while traversing all data in fp32_dir and fq_dir
    def advance_on(self, fp32_dir, fq_dir):
        data_paths = self.collect_data_path(fp32_dir, fq_dir)
        for tensor_name, data_path in data_paths.items():
            for (fp32_data_path, fq_data_path) in data_path:
                fp32_data = np.load(fp32_data_path)
                fq_data = np.load(fq_data_path)

                MSE = np.square(fp32_data - fq_data).mean()

                self.qerror_map[tensor_name] += MSE

                self.qerror_min = min(MSE, self.qerror_min)
                self.qerror_max = max(MSE, self.qerror_max)

    # Return
    # qerror_map (dict: tensor_name(string) -> qerror(float))
    # qerror_min (float)
    # qerror_max (float)
    def get_final_result(self):
        qerror_map = dict()
        for tensor_name, acc in self.qerror_map.items():
            qerror_map[tensor_name] = acc / self._num_processed_data

        return qerror_map, self.qerror_min, self.qerror_max

    def run(self):
        self.advance_on(self._fp32_dir, self._fq_dir)
        return self.get_final_result()


class TAEComputer(QErrorComputer):  #total absolute error
    def __init__(self, fp32_dir, fq_dir):
        super().__init__(fp32_dir, fq_dir)
        self.total_error = 0
        self.qerror_min = float('inf')
        self.qerror_max = -self.qerror_min

    def advance_on(self, fp32_dir, fq_dir):
        data_paths = self.collect_data_path(fp32_dir, fq_dir)
        for tensor_name, data_path in data_paths.items():
            for (fp32_data_path, fq_data_path) in data_path:
                fp32_data = np.load(fp32_data_path)
                fq_data = np.load(fq_data_path)

                total_error = np.sum(np.abs(fp32_data - fq_data))

                self.qerror_map[tensor_name] += total_error

                self.qerror_min = min(total_error, self.qerror_min)
                self.qerror_max = max(total_error, self.qerror_max)

    # Return
    # qerror_map (dict: tensor_name(string) -> qerror(float))
    # qerror_min (float)
    # qerror_max (float)
    def get_final_result(self):
        qerror_map = dict()
        for tensor_name, acc in self.qerror_map.items():
            qerror_map[tensor_name] = acc / self._num_processed_data
        return qerror_map, self.qerror_min, self.qerror_max

    def run(self):
        self.advance_on(self._fp32_dir, self._fq_dir)
        return self.get_final_result()


# Scaled Root Mean Square Error (SRMSE)
# SRMSE = sqrt(MSE) / scale
class SRMSEComputer(QErrorComputer):
    def __init__(self, fp32_dir, fq_dir):
        super().__init__(fp32_dir, fq_dir)
        if fq_dir != None:
            self.scale_file = Path(fq_dir) / 'scales.txt'

    # Incrementally compute Qerror while traversing all data in fp32_dir and fq_dir
    def advance_on(self, fp32_dir, fq_dir):
        if fq_dir != None:
            self.scale_file = Path(fq_dir) / 'scales.txt'
            self._fq_dir = fq_dir

        data_paths = self.collect_data_path(fp32_dir, fq_dir)

        for tensor_name, data_path in data_paths.items():
            for (fp32_data_path, fq_data_path) in data_path:
                fp32_data = np.load(fp32_data_path)
                fq_data = np.load(fq_data_path)

                MSE = np.square(fp32_data - fq_data).mean()

                self.qerror_map[tensor_name] += MSE

    # Return
    # qerror_map (dict: tensor_name(string) -> qerror(float))
    # qerror_min (float)
    # qerror_max (float)
    def get_final_result(self):
        with open(self.scale_file) as f:
            # scale_map: {tensor_name(str) -> scale(float)}
            scale_map = json.load(f)

        qerror_max = 0.0
        qerror_map = dict()
        for tensor_name, acc in self.qerror_map.items():
            MSE = acc / self._num_processed_data
            SRMSE = np.sqrt(MSE) / scale_map[tensor_name]
            qerror_map[tensor_name] = SRMSE
            qerror_max = max(SRMSE, qerror_max)

        return qerror_map, 0.0, qerror_max

    def run(self):
        self.advance_on(self._fp32_dir, self._fq_dir)
        return self.get_final_result()
