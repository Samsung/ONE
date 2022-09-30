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

import h5py as h5
import numpy as np
import argparse

from circle.Model import Model
from RandomDataGenerator import RandomDataGenerator

#
# This script generates a pack of random input data (.h5) expected by the input circle model
#
# Basic usage:
#   gen_h5_random_inputs.py --model <path/to/circle/model> --num_data <number/of/data> --output <path/to/output/data>
#   ex: gen_h5_random_inputs.py --model add.circle --num_data 3 --output add.circle.input.h5
#   (This will create add.circle.input.h5 composed of three random inputs in the same directory as the model)
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num_data', type=int, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

model = args.model
num_data = args.num_data
output_path = args.output

with open(model, 'rb') as f:
    buf = f.read()
    circle_model = Model.GetRootAsModel(buf, 0)

# Assume one subgraph
assert (circle_model.SubgraphsLength() == 1)
graph = circle_model.Subgraphs(0)
inputs = graph.InputsAsNumpy()

# Create h5 file
h5_file = h5.File(output_path, 'w')
group = h5_file.create_group("value")
group.attrs['desc'] = "Input data for " + model

# Generate random data
for i in range(num_data):
    sample = group.create_group(str(i))
    for j in range(len(inputs)):
        input_index = inputs[j]
        tensor = graph.Tensors(input_index)
        g = RandomDataGenerator(tensor.ShapeAsNumpy())
        input_data = g.gen(tensor.Type())
        sample.create_dataset(str(j), data=input_data)

h5_file.close()
