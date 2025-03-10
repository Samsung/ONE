#!/usr/bin/env python3

import sys
import h5py

from pathlib import Path

# local python files
import util_h5_file


# convert() will generate exec_circle input data files from h5 file from exec_onnx
def convert(ref_model, circle_model, is_nchw=False):
    ref_model_name = Path(ref_model).name
    input_data_path = util_h5_file.get_input_filename(ref_model_name, is_nchw)
    h5f = h5py.File(input_data_path, 'r')
    input_names = []
    input_data = dict()
    for t in h5f:
        input_name = str(t)
        input_names.append(input_name)
        if h5f[input_name].shape == ():
            input_data[input_name] = h5f[input_name][()]
        else:
            input_data[input_name] = h5f[input_name][:]

    h5f.close()

    # write numinputs file
    num_inputs = len(input_names)
    num_inputs_file = open(circle_model + '.numinputs', 'w')
    num_inputs_file.write(str(num_inputs))
    num_inputs_file.close()

    # write input data file for exec_circle
    for i in range(num_inputs):
        input_name = input_names[i]
        input_data[input_name].tofile(circle_model + '.input' + str(i))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        filepath = Path(sys.argv[0])
        sys.exit('Usage: ' + filepath.name + ' [model.reference] [model.cirle]')

    convert(sys.argv[1], sys.argv[2], True)
