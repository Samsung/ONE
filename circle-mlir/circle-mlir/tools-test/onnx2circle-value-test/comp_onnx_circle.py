#!/usr/bin/env python3

import sys
import numpy as np
import h5py

from pathlib import Path

# local python files
import util_h5_file
import util_validation


def compare_outputs(onnx_model,
                    circle_model,
                    rtolerance,
                    atolerance,
                    o_name2idx_map,
                    is_nchw=False):
    # restore onnx h5 output
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

    # restore circle output
    circle_output_data = dict()
    for idx in range(len(output_names)):
        output_name = output_names[idx]
        output_np_type = onnx_output_data[output_name].dtype
        output_np_shape = onnx_output_data[output_name].shape

        file_idx = idx
        if o_name2idx_map != None:
            file_idx = o_name2idx_map[output_name]

        output_np_data = np.fromfile(circle_model + '.output' + str(file_idx),
                                     output_np_type)
        circle_output_data[output_name] = np.reshape(output_np_data, output_np_shape)

    result_compare = True
    scalar_exist = False
    peir_results = dict()

    for idx in range(len(output_names)):
        output_name = output_names[idx]
        onnx_output = onnx_output_data[output_name]
        circle_output = circle_output_data[output_name]

        if circle_output.reshape(-1).shape[0] == 1:
            scalar_exist = True
            print('Output tensor `{}` is scalar. Skip circle validation.'.format(
                output_name))
            peir_results['output {}'.format(idx)] = 'N/A'
            continue

        diff = np.isclose(onnx_output, circle_output, rtol=rtolerance, atol=atolerance)

        result_compare_one = np.all(diff)
        print('Compare', idx, result_compare_one)
        if (not result_compare_one):
            diff_val = np.subtract(onnx_output, circle_output)
            print('ONNX Result', onnx_output)
            print('Diff', diff_val)
            print('Diff Max', np.max(diff_val), 'in tolerance', rtolerance, atolerance)

        result_compare = result_compare and result_compare_one

        peir_result = util_validation.get_peir_from(onnx_output, circle_output)
        peir_results['output {}'.format(idx)] = peir_result

    save_res = util_validation.save_output_peir_to_json(peir_results, circle_model)

    if (scalar_exist):
        return 0

    if (not result_compare or not save_res):
        return 1

    return 0


if __name__ == '__main__':
    rtolerance_def = 1e-04
    atolerance_def = 1e-04
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        filepath = Path(sys.argv[0])
        sys.exit('Usage: ' + filepath.name + ' [model.onnx] [model.cirle] (rtol atol)')

    if len(sys.argv) == 5:
        rtolerance_def = float(sys.argv[3])
        atolerance_def = float(sys.argv[4])

    res = compare_outputs(sys.argv[1],
                          sys.argv[2],
                          rtolerance=rtolerance_def,
                          atolerance=atolerance_def,
                          o_name2idx_map=None,
                          is_nchw=True)
    sys.exit(res)
