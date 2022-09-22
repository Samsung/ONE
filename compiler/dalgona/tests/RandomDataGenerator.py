#!/usr/bin/env python3

# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from circle.TensorType import TensorType


class RandomDataGenerator:
    def __init__(self, shape):
        self.shape = shape

    def _unsupported_types(self, dtype):
        raise RuntimeError('Unsupported data type')

    def _gen_uint8(self, dtype):
        return np.random.randint(0, high=256, size=self.shape, dtype=np.uint8)

    def _gen_int16(self, dtype):
        return np.random.randint(-32767, high=32768, size=self.shape, dtype=np.int16)

    def _gen_float32(self, dtype):
        return np.array(10 * np.random.random_sample(self.shape) - 5, np.float32)

    def gen(self, dtype):
        gen_book = dict()
        gen_book[TensorType.UINT8] = self._gen_uint8
        gen_book[TensorType.INT16] = self._gen_int16
        gen_book[TensorType.FLOAT32] = self._gen_float32

        return gen_book.get(dtype, self._unsupported_types)(dtype)
