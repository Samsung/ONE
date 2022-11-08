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
'''Test visqlib.QErrorComputer module'''

import unittest
import tempfile
import numpy as np
import os

from visqlib.QErrorComputer import MPEIRComputer
from visqlib.QErrorComputer import MSEComputer


class VisqQErrorComputerTest(unittest.TestCase):
    def setUp(self):
        "Called before running each test"
        self.fp32_dir = tempfile.TemporaryDirectory()
        self.fq_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        "Called after running each test"
        self.fp32_dir.cleanup()
        self.fq_dir.cleanup()

    def _setUpSingleTensorData(self):
        with open(self.fp32_dir.name + '/tensors.txt', 'w') as f:
            f.write('test')
        with open(self.fq_dir.name + '/tensors.txt', 'w') as f:
            f.write('test')
        os.mkdir(self.fp32_dir.name + '/0')
        os.mkdir(self.fq_dir.name + '/0')
        test_data = np.zeros(16)
        np.save(self.fp32_dir.name + '/0/test.npy', test_data)
        np.save(self.fq_dir.name + '/0/test.npy', test_data)

    def _setUpDifferentTensorData(self):
        # Two fp32 data (test, test2)
        # One fq data (test)
        # NOTE When does this happen?
        # This case can happen because visq ignores nodes that do not affect qerrors.
        # For example, RESHAPE Op does not affect qerrors, so its fq data is not dumped,
        # although it is listed in 'tensors.txt'.
        with open(self.fp32_dir.name + '/tensors.txt', 'w') as f:
            f.writelines(['test\n', 'test2'])
        with open(self.fq_dir.name + '/tensors.txt', 'w') as f:
            f.writelines(['test\n', 'test2'])
        os.mkdir(self.fp32_dir.name + '/0')
        os.mkdir(self.fq_dir.name + '/0')
        test_data = np.zeros(16)
        np.save(self.fp32_dir.name + '/0/test.npy', test_data)
        np.save(self.fp32_dir.name + '/0/test2.npy', test_data)
        np.save(self.fq_dir.name + '/0/test.npy', test_data)

    def test_MPEIR(self):
        self._setUpSingleTensorData()

        computer = MPEIRComputer(self.fp32_dir.name, self.fq_dir.name)
        qmap = computer.run()
        self.assertAlmostEqual(0.0, qmap['test'])

    def test_MPEIR_different_tensors(self):
        self._setUpDifferentTensorData()

        computer = MPEIRComputer(self.fp32_dir.name, self.fq_dir.name)
        qmap = computer.run()
        self.assertAlmostEqual(0.0, qmap['test'])

    def test_MSE(self):
        self._setUpSingleTensorData()

        computer = MSEComputer(self.fp32_dir.name, self.fq_dir.name)
        qmap, qmin, qmax = computer.run()
        self.assertAlmostEqual(0.0, qmap['test'])
        self.assertAlmostEqual(0.0, qmin)
        self.assertAlmostEqual(0.0, qmax)

    def test_MSE_different_tensors(self):
        self._setUpDifferentTensorData()

        computer = MSEComputer(self.fp32_dir.name, self.fq_dir.name)
        qmap, qmin, qmax = computer.run()
        self.assertAlmostEqual(0.0, qmap['test'])
        self.assertAlmostEqual(0.0, qmin)
        self.assertAlmostEqual(0.0, qmax)


if __name__ == '__main__':
    unittest.main()
