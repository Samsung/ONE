#!/usr/bin/env python3

import h5py


def get_input_filename(model_name, is_nchw=False):
    if is_nchw:
        return model_name + '.nchw.input.h5'
    else:
        return model_name + '.nhwc.input.h5'


def get_output_filename(model_name, is_nchw=False):
    if is_nchw:
        return model_name + '.nchw.output.h5'
    else:
        return model_name + '.nhwc.output.h5'


def get_h5_from(file_path, mode='r'):
    if mode != 'r' and mode != 'w':
        print('Invalid file mode for get_h5_from')
        raise

    try:
        h5 = h5py.File(file_path, mode)
    except:
        print('cannot open {}. Please check the file path.'.format(file_path))
        raise

    return h5
