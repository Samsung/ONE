#!/usr/bin/env python3
import h5py as h5
import numpy as np
from circle.Model import Model
from circle.TensorType import TensorType
import argparse


def toNumpyType(circle_type):
    if circle_type == TensorType.UINT8:
        return np.uint8
    if circle_type == TensorType.FLOAT32:
        return np.float32
    if circle_type == TensorType.INT16:
        return np.int16


#
# This script generates a pack of random input data (.h5) expected by the input tflite model
#
# Basic usage:
#   gen_h5_random_inputs.py --model <path/to/circle/model> --num_data <number/of/data> --output <path/to/output/data>
#   ex: gen_h5_random_inputs.py --model add.tflite --num_data 3 --output add.tflite.input.h5
#   (This will create add.tflite.input.h5 composed of three random inputs in the same directory as the model)
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
    sample.attrs['desc'] = "Input data " + str(i)

    for j in range(len(inputs)):
        input_index = inputs[j]
        tensor = graph.Tensors(input_index)
        np_type = toNumpyType(tensor.Type())
        if np_type == np.uint8:
            input_data = np.random.randint(
                0, high=256, size=tensor.ShapeAsNumpy(), dtype=np_type)
        elif np_type == np.float32:
            input_data = np.array(10 * np.random.random_sample(tensor.ShapeAsNumpy()) - 5,
                                  np_type)
        elif np_type == np.int16:
            input_data = np.random.randint(
                -32767, high=32768, size=tensor.ShapeAsNumpy(), dtype=np_type)
        else:
            raise SystemExit('Unsupported data type')
        sample.create_dataset(str(j), data=input_data)

h5_file.close()
