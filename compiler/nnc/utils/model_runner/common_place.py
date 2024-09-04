import h5py
import argparse
from argparse import RawTextHelpFormatter


def regular_step():
    """
    This function is intended to decompose the necessary steps to obtain information from the command line.

    :return: argparse object, which hold paths to nn model and input data
    """
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('-m',
                        '--model',
                        help=("specify input file with NN model, \n[depends from model, "
                              " two for caffe and caffe2, one for onnx and tflite]"),
                        nargs='+')
    parser.add_argument('-i',
                        '--input',
                        help=(" specify file with neural"
                              " network input data, hdf5 for caffe caffe2 tflite "
                              "and pb for onnx"),
                        required=True)
    parser.add_argument(
        '-o',
        '--output_path',
        help='here you specify which place will hold your output, default here',
        default='')

    args = parser.parse_args()
    # added to check is our input file or not. most simple way
    try:
        with open(args.input) as f:
            pass
    except IOError as e:
        print('input file your enter doesnt exist!')

    # added to check is our model right or not
    try:
        for i in args.model:
            with open(i) as f:
                pass
    except IOError as e:
        print('model you enter doesnt exist, write correct PATH ')

    return args


def save_result(output_path, output_data):
    """
    This function save result of nn working in .hdf5 file
    :param output_path: you specify directory to store your result
    :param output_data: information that you write to .hdf5 file
    :return:
    """
    with open(output_path + 'responce.txt', 'w+') as f:
        f.write(str(output_data))
    f = h5py.File(output_path + 'responce.hdf5', 'w')
    f.create_dataset('out', dtype='float32', data=output_data)
    f.close()


def read_input(input_path):
    h5f = h5py.File(input_path, 'r')
    for t in h5f:
        tensorName = str(t)
    return h5py.File(input_path, 'r')[tensorName][:]
