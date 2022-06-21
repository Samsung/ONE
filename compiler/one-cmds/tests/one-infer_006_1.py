# This script converts one set of I/O npy data files to one h5 file

import glob
import h5py as h5
import numpy as np
import sys
from pathlib import Path

def npy2h5(npy_prefix, h5_name):
    h5File = h5.File(h5_name, mode='w')

    input_npy_glob = Path(npy_prefix + ".input.*.npy")
    input_npy_list = glob.glob(str(input_npy_glob))

    if len(input_npy_list) == 0:
        sys.exit("Input NPY is not found")
    else:
        h5File.create_group('input')

    for idx, npy_file in enumerate(sorted(input_npy_list, key=lambda k: k.lower())):
        npyData = np.load(npy_file)
        dtype = npyData.dtype
        shape = npyData.shape

        if dtype not in [np.float32, np.uint8]:
            sys.exit("Input dtype is not supported")

        h5File.create_dataset(f'input/{str(idx)}', shape, dtype, npyData)

    output_npy_glob = Path(npy_prefix + ".output.*.npy")
    output_npy_list = glob.glob(str(output_npy_glob))

    if len(output_npy_list) == 0:
        sys.exit("Output NPY is not found")
    else:
        h5File.create_group('output')

    for idx, npy_file in enumerate(sorted(output_npy_list, key=lambda k: k.lower())):
        npyData = np.load(npy_file)
        dtype = npyData.dtype
        shape = npyData.shape

        if dtype not in [np.float32, np.uint8]:
            sys.exit("Output dtype is not supported")

        h5File.create_dataset(f'output/{str(idx)}', shape, dtype, npyData)

    print(f"{h5_name} has been successfully generated")


def main():
    if len(sys.argv) < 3:
        filepath = Path(sys.argv[0])
        sys.exit("Usage: " + filepath.name + " [NPY Prefix] [H5 File]")
    
    npy_prefix = sys.argv[1]
    h5_name = sys.argv[2]

    npy2h5(npy_prefix, h5_name)


if __name__ == '__main__':
    main()
