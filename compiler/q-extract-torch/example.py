import os.path

import torch
from torch.ao.quantization import QConfig, PerChannelMinMaxObserver

from Torch_Circle_Mapper import Torch2CircleMapper
from Torch_Extractor import TorchExtractor

model_name = "mobilenet_v2"
model = torch.hub.load("pytorch/vision:v0.14.1", model_name,  pretrained=True)

if not os.path.exists(model_name):
    os.makedirs(model_name)

input = torch.randn(1, 3, 224, 224)

mapper = Torch2CircleMapper(original_model=model, sample_input=input, dir_path=model_name)
mapping, data = mapper.get_mapped_dict()


model.eval()
model.qconfig = QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
                        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8))
p_model = torch.quantization.prepare(model)
p_model(input)
quant = torch.quantization.convert(p_model)

extractor = TorchExtractor(quant, json_path=model_name + '/qparam.json', qdtype=torch.quint8, partial_graph_data=data, mapping=mapping)
extractor.generate_files()
