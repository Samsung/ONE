#!/usr/bin/env python3
# Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

# Library to compute the parameters for feature map equalizer

import numpy as np

from typing import List, Tuple

Tensors = List[np.ndarray]


def _channelwiseMinMax(tensors: Tensors, channel: int) -> Tuple[List[float], List[float]]:
    """
    Compute channel-wise min and max for the tensor.
    :param tensors: a list of numpy array (each is a tensor)
    :param channel: number of channels
    :return: lists of min and max for each channel
    """
    channel_wise_min = []
    channel_wise_max = []
    for c in range(channel):
        min_act = min(tensors, key=lambda activation: np.min(activation[:, :, :, c]))
        max_act = max(tensors, key=lambda activation: np.max(activation[:, :, :, c]))
        channel_wise_min.append(float(np.min(min_act[:, :, :, c])))
        channel_wise_max.append(float(np.max(max_act[:, :, :, c])))
    return channel_wise_min, channel_wise_max


def getActivationMax(tensor: np.ndarray) -> np.ndarray:
    """
    Get max values of activation.
    :param tensors: a list of numpy array (each is a tensor)
    :return: 1D array of max absolute values for each channel
    """
    # max along with last dimension
    return np.abs(tensor.reshape(-1, tensor.shape[-1])).max(axis=0)
