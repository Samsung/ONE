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
import numpy as np
import collections
import json
import torch.nn.quantized.modules.linear


def quantize_tensor(tensor: torch.Tensor, scale, zero_point,
                    dtype=np.int8) -> torch.Tensor:
    if dtype not in (np.uint8, np.int8, np.int32):
        raise Exception('Please check dtype')
    new_tensor = tensor.clone().detach().numpy()
    new_tensor = new_tensor / scale + zero_point
    return new_tensor.astype(dtype)


class TorchExtractor:
    qdtype_mapping = {
        torch.quint8: {
            'str': "uint8",
            'np': np.uint8
        },
        torch.qint8: {
            'str': "int8",
            'np': np.int8
        },
        torch.qint32: {
            'str': "int32",
            'np': np.int32
        }
    }

    @staticmethod
    def permute(tensor: torch.Tensor) -> torch.Tensor:
        dim = len(tensor.shape)
        if dim == 4:  # NCHW to NHWC
            tensor = tensor.permute(0, 2, 3, 1)
        return tensor

    def __init__(self, quantized_model: torch.nn.Module, json_path: str,
                 partial_graph_data: None):
        self.__np_idx = 0
        self.__input_dtype = None
        self.__graph_data = collections.OrderedDict()
        self.__partial_graph_data = partial_graph_data
        self.__json_path = json_path
        self.__dir_path, self.__json_file_name = os.path.split(json_path)
        self.__extract_module(quantized_model)

    def __extract_module(self, module: torch.nn.Module):
        graph_data = self.__graph_data
        partial_graph_data = self.__partial_graph_data
        # Restructuring Neural Network model
        for name, mod in module.named_modules():
            # TODO: check whether there is better way to check instance of \
            #  torch.nn.quantized.modules.* and not torch.nn.modules.Module
            """
            Need to skip just Module. Only Operator/Tensor/Activation Needed
            When just using 'isinstance', all of operator/tensor/activation belong to it
            (All of them inherit torch.nn.modules.Module)

            Why '.nn.quantized.modules' instead of 'torch.nn.quantized.modules'?
            On torch 1.7.0, the path is 'torch.nn.quantized.modules',
            But on latest version, the path is 'torch.ao.nn.quantized.modules'
            """
            if name == '' or str(type(mod)).find('.nn.quantized.modules') == -1:
                continue
            if isinstance(mod, torch.nn.quantized.modules.linear.LinearPackedParams):
                continue

            if self.__input_dtype is None and hasattr(mod, 'scale') and hasattr(
                    mod, 'zero_point'):
                self.__input_dtype = mod.dtype

            if name in graph_data:
                data = graph_data[name]
            elif name in partial_graph_data:
                data = graph_data[name] = partial_graph_data[name]
            else:
                data = {}
                graph_data[name] = data
            for value_name, tensor in mod.state_dict().items():
                # Need to skip just Module. Only Operator/Tensor/Activation Needed
                # TODO: Find better way to check instance of torch.nn.quantized.modules
                if str(type(mod)).find('.nn.quantized.modules') == -1:
                    continue
                tensor_name = value_name[value_name.rfind(".") + 1:]
                prefix = value_name[:value_name.rfind(".")]
                # for Linear
                if prefix.find('_packed_params') != -1:
                    if tensor_name == '_packed_params':
                        data['weight'] = tensor[0]
                        data['bias'] = tensor[1]
                    continue

                data[tensor_name] = TorchExtractor.permute(tensor)

    def __save_np(self, data):
        file_name = str(self.__np_idx) + ".npy"
        if data.shape == ():
            data = np.array([data])
        if data.dtype == np.dtype(np.float64):
            data = data.astype(np.float32)
        np.save(os.path.join(self.__dir_path, file_name), data)
        self.__np_idx += 1
        return file_name

    def __from_tensor(self, tensor):
        if tensor is None:
            raise Exception('tensor is null')
        data = {}
        if tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            data['scale'] = self.__save_np(np.array(tensor.q_scale()))
            data['zerop'] = self.__save_np(np.array(tensor.q_zero_point()))
            data['quantized_dimension'] = 0
        elif tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric,
                                  torch.per_channel_affine_float_qparams):
            data['scale'] = self.__save_np(tensor.q_per_channel_scales().numpy())
            data['zerop'] = self.__save_np(tensor.q_per_channel_zero_points().numpy())
            data['quantized_dimension'] = tensor.q_per_channel_axis()

        if tensor.dtype == torch.qint8:
            data['value'] = self.__save_np(torch.int_repr(tensor).numpy())
        else:
            data['value'] = self.__save_np(tensor.numpy())
        data['dtype'] = self.qdtype_mapping[tensor.dtype]['str']
        return data

    def generate_files(self, mapping: None):
        graph_data = self.__graph_data
        mapped_data = {}
        not_mapped_data = {}
        if not os.path.exists(self.__dir_path):
            os.makedirs(self.__dir_path, exist_ok=True)

        # method should work even there is no mapping data => all data will be not_mapped_data
        if mapping is None:
            mapping = {}

        for name, layer in graph_data.items():
            dtype = self.qdtype_mapping[self.__input_dtype]['str']
            if "weight" in layer:
                w_name = name + '.weight'
                tensor = layer['weight']
                if w_name in mapping:
                    data = mapped_data
                    w_name = mapping[w_name]
                else:
                    data = not_mapped_data
                if tensor.is_quantized:
                    data[w_name] = self.__from_tensor(tensor=tensor)
            if "scale" in layer and "zero_point" in layer:
                scale = layer['scale'].numpy()
                zero_point = layer['zero_point'].numpy()

                layer_name = name
                if layer_name in mapping:
                    layer_name = mapping[layer_name]
                    data = mapped_data
                else:
                    data = not_mapped_data

                s_np = self.__save_np(scale)
                z_np = self.__save_np(zero_point)
                data[layer_name] = {
                    'scale': s_np,
                    'zerop': z_np,
                    'dtype': dtype,
                    'quantized_dimension': 0
                }

                b_name = name + '.bias'
                if b_name in mapping:
                    b_name = mapping[b_name]
                    data = mapped_data
                else:
                    data = not_mapped_data

                if "bias" in layer:
                    quantized_bias = quantize_tensor(
                        layer['bias'], scale, zero_point, dtype=np.int32)
                    data[b_name] = {
                        'scale': s_np,
                        'zerop': z_np,
                        'dtype': 'int32',
                        'value': self.__save_np(quantized_bias),
                        'quantized_dimension': 0
                    }
            # such as RELU or transpose like that, inherit quantization parameter
            elif 'prev_op' in layer:
                parent_name = graph_data[name]['prev_op']
                if mapping[parent_name] in mapped_data:
                    parent = mapped_data[mapping[parent_name]]
                else:
                    parent = not_mapped_data[parent_name]

                if parent_name + '.out' in mapping:
                    t_name = mapping[parent_name + '.out']
                    data = mapped_data
                else:
                    t_name = name
                    data = not_mapped_data

                data[t_name] = {
                    'scale': parent['scale'],
                    'zerop': parent['zerop'],
                    'dtype': parent['dtype'],
                    'quantized_dimension': 0
                }
        with open(self.__json_path, 'w') as json_file:
            json.dump(mapped_data, json_file)
        if len(not_mapped_data) > 0:
            not_mapped_path = os.path.join(self.__dir_path,
                                           'not_mapped_' + self.__json_file_name)
            with open(not_mapped_path, 'w') as json_file:
                json.dump(not_mapped_data, json_file)
