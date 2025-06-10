#!/usr/bin/env python3

import sys
import numpy as np
import h5py

from pathlib import Path

# local python files
import util_h5_file


def compare_outputs(model_name, rtolerance, atolerance, is_nchw=False):
    onnx_model = model_name + '.onnx'
    circle_src_model = model_name + '.circle'
    circle_two_model = model_name + '.2.circle'

    # to compare two circle outputs, we need to get dtype and number of elements
    # of each outputs to compare in meaningful way.
    # to do this, we fetch these from onnx model outputs and store them in
    # onnx_output_data dict.

    # restore onnx h5 output
    # onnx h5 file is used to collect output names
    onnx_model_name = Path(onnx_model).name
    output_data_path = util_h5_file.get_output_filename(onnx_model_name, is_nchw)
    h5f = h5py.File(output_data_path, 'r')
    output_names = []
    onnx_output_data = dict()
    for t in h5f:
        output_name = str(t)
        output_names.append(output_name)
        h5_output = h5f[output_name]
        if h5_output.shape == ():
            # https://github.com/h5py/h5py/issues/1779#issuecomment-743447638
            onnx_output_data[output_name] = h5_output[()]
        else:
            onnx_output_data[output_name] = h5_output[:]

    h5f.close()

    # restore circle outputs
    circle_src_output_data = dict()
    circle_two_output_data = dict()
    for idx in range(len(output_names)):
        output_name = output_names[idx]
        output_np_type = onnx_output_data[output_name].dtype
        output_np_shape = onnx_output_data[output_name].shape

        output_np_data = np.fromfile(circle_src_model + '.output' + str(idx),
                                     output_np_type)
        circle_src_output_data[output_name] = np.reshape(output_np_data, output_np_shape)

        output_np_data = np.fromfile(circle_two_model + '.output' + str(idx),
                                     output_np_type)
        circle_two_output_data[output_name] = np.reshape(output_np_data, output_np_shape)

    result_compare = True

    for idx in range(len(output_names)):
        output_name = output_names[idx]
        circle_src_output = circle_src_output_data[output_name]
        circle_two_output = circle_two_output_data[output_name]

        diff = np.isclose(circle_src_output,
                          circle_two_output,
                          rtol=rtolerance,
                          atol=atolerance)

        result_compare_src_two = np.all(diff)
        print('Compare', idx, result_compare_src_two)
        if (not result_compare_src_two):
            diff_val = np.subtract(circle_src_output, circle_two_output)
            print('SRC Result', circle_src_output)
            print('Diff', diff_val)
            print('Diff Max', np.ndarray.max(diff_val), 'in tolerance', rtolerance,
                  atolerance)

        result_compare = result_compare and result_compare_src_two

    return 0 if result_compare else 1


if __name__ == '__main__':
    rtolerance_def = 1e-04
    atolerance_def = 1e-04
    if len(sys.argv) != 2 and len(sys.argv) != 4:
        filepath = Path(sys.argv[0])
        sys.exit('Usage: ' + filepath.name + ' [model] (rtol atol)')

    if len(sys.argv) == 4:
        rtolerance_def = float(sys.argv[2])
        atolerance_def = float(sys.argv[3])

    res = compare_outputs(sys.argv[1],
                          rtolerance=rtolerance_def,
                          atolerance=atolerance_def,
                          is_nchw=True)
    sys.exit(res)
