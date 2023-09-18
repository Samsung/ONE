# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

import torch
import torch.nn
import torch.quantization

from Torch_Circle_Mapper import Torch2CircleMapper
from TorchExtractor import TorchExtractor


# Helper class of PyTorch Quantization Parameter Export
class TorchQParamExporter:
    @staticmethod
    def export(original_model: torch.nn.Module,
               quantized_model: torch.nn.Module,
               sample_input: torch.tensor,
               json_path: str,
               tflite2circle_path='tflite2circle'):
        if original_model is None or not isinstance(original_model, torch.nn.Module):
            raise Exception("There is no original Pytorch Model")
        if quantized_model is None or not isinstance(quantized_model, torch.nn.Module):
            raise Exception("There is no quantized Pytorch Model")
        if json_path is None:
            raise Exception("Please specify save path")
        if sample_input is None or not isinstance(sample_input, torch.Tensor):
            raise Exception("Please give sample input of network")
        dir_path = os.path.dirname(json_path)
        mapper = Torch2CircleMapper(
            original_model=original_model,
            sample_input=sample_input,
            dir_path=dir_path,
            tflite2circle_path=tflite2circle_path,
            clean_circle=False)
        mapping, partial_graph_data = mapper.get_mapped_dict()
        extractor = TorchExtractor(
            quantized_model=quantized_model,
            json_path=json_path,
            partial_graph_data=partial_graph_data)
        extractor.generate_files(mapping)
