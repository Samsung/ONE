#!/usr/bin/env python3
import h5py as h5
import numpy as np
from circle.Model import Model
from circle.TensorType import TensorType
import argparse
import glob
import os


def chunks(lst, n):
    """Yield successive n-sized chunks from the list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def toNumpyType(circle_type):
    if circle_type == TensorType.UINT8:
        return np.uint8
    if circle_type == TensorType.FLOAT32:
        return np.float32
    if circle_type == TensorType.INT16:
        return np.int16


#
# This script generates a pack of random input data (.h5) expected by the input circle models
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
parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='Output directory where the inputs are generated')
parser.add_argument(
    '--artifact_dir',
    type=str,
    required=True,
    help='Artifact directory where test files exist')
parser.add_argument(
    '--input_dir',
    type=str,
    required=True,
    help='Input directory where input text files exist')
parser.add_argument(
    '--test_param',
    type=str,
    required=True,
    nargs='+',
    help=
    'All the param list to test. e.g. ${RECIPE_NAME_0} ${GRANULARITY_0} ${DTYPE_0} ${RECIPE_NAME_1} ${GRANULARITY_1} ${DTYPE_1}..'
)
parser.add_argument('--config', action='store_true', help='Generate inputs with config')
parser.add_argument('--mode', type=str, default='default', help='Mode to test')
args = parser.parse_args()

output_dir = args.output_dir
artifact_dir = args.artifact_dir
input_dir = args.input_dir
test_param = args.test_param
config = args.config
mode = args.mode

modes_to_input_h5_suffix = {
    'default': 'input.h5',
    'mixed_quantization': 'mixed.input.h5',
}

test_params = test_param[0].split()
PARAM_SET_SIZE = 3
assert (len(test_params) % PARAM_SET_SIZE) == 0
test_params = list(chunks(test_params, PARAM_SET_SIZE))
for idx in range(len(test_params)):
    model_name = test_params[idx][0]
    granularity = test_params[idx][1]
    dtype = test_params[idx][2]

    model = os.path.join(artifact_dir, model_name + '.circle')
    with open(model, 'rb') as f:
        buf = f.read()
        circle_model = Model.GetRootAsModel(buf, 0)

    # Assume one subgraph
    assert (circle_model.SubgraphsLength() == 1)
    graph = circle_model.Subgraphs(0)
    inputs = graph.InputsAsNumpy()

    testcase = f'{model_name}.{granularity}.{dtype}'
    output = os.path.join(output_dir, f'{testcase}.{modes_to_input_h5_suffix[mode]}')

    # Create h5 file
    h5_file = h5.File(output, 'w')
    group = h5_file.create_group("value")
    group.attrs['desc'] = "Input data for " + model

    if config:
        input_text_dir = os.path.join(input_dir,
                                      f'{model_name}_config/{granularity}/{dtype}')
    else:
        input_text_dir = os.path.join(input_dir, f'{model_name}/{granularity}/{dtype}')
    # Input files
    records = sorted(glob.glob(input_text_dir + "/*.txt"))
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
