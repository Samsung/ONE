#!/usr/bin/env python3
import h5py as h5
import numpy as np
from circle.Model import Model
from circle.TensorType import TensorType
import argparse
import glob

#
# This script generates a pack of random input data (.h5) expected by the input circle model
#
# Basic usage:
#   gen_h5_explicit_inputs.py --model <path/to/model/file> --input <path/to/input/directory> --output <path/to/output/file>
#   ex: gen_h5_explicit_inputs.py --model Add_000.circle --input Add_000 --output Add_000.input.h5
#   (This will create Add_000.input.h5)
#
# The input directory should be organized as follows
# <input_directory>/
#   -> <record_index>.txt
#     ...
# Each txt file has the explicit values of inputs
# Example. if the model has two inputs whose shapes are both (1, 3),
# the first record file name is 0.txt, and its contents is something like below
# 1, 2, 3
# 4, 5, 6
#
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

model = args.model
input = args.input
output = args.output

with open(model, 'rb') as f:
    buf = f.read()
    circle_model = Model.GetRootAsModel(buf, 0)

# Assume one subgraph
assert (circle_model.SubgraphsLength() == 1)
graph = circle_model.Subgraphs(0)
inputs = graph.InputsAsNumpy()

# Create h5 file
h5_file = h5.File(output, 'w')
group = h5_file.create_group("value")
group.attrs['desc'] = "Input data for " + model


def toNumpyType(circle_type):
    if circle_type == TensorType.UINT8:
        return np.uint8
    if circle_type == TensorType.FLOAT32:
        return np.float32
    if circle_type == TensorType.INT16:
        return np.int16


# Input files
records = sorted(glob.glob(input + "/*.txt"))
for i, record in enumerate(records):
    sample = group.create_group(str(i))
    sample.attrs['desc'] = "Input data " + str(i)
    with open(record, 'r') as f:
        lines = f.readlines()
        for j, line in enumerate(lines):
            data = np.array(line.split(','))
            input_index = inputs[j]
            tensor = graph.Tensors(input_index)
            np_type = toNumpyType(tensor.Type())
            input_data = np.array(data.reshape(tensor.ShapeAsNumpy()), np_type)
            sample.create_dataset(str(j), data=input_data)

h5_file.close()
