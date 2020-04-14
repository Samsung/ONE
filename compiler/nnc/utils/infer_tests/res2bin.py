import numpy as np
import h5py
import struct
import sys

# This script takes hdf5 file and unfolds it in a vector of float values
# which is then writen in binary format to a given file
# This is used by infer_testcases.py


def res2bin(infilename, outfilename):
    # print("Input filename: ", infilename)
    # print("Output filename: " , outfilename)

    f = h5py.File(infilename)
    dset = f[list(f.keys())[0]]

    vals = np.zeros(np.shape(dset), dtype='float32')
    for i in range(np.size(dset, 0)):
        vals[i, :] = np.asarray(dset[i], dtype='float32')
    vals = list(np.reshape(vals, (vals.size)))

    with open(outfilename, 'wb') as outfile:
        outfile.write(struct.pack('f' * len(vals), *vals))


if __name__ == '__main__':
    argc = len(sys.argv)
    if (argc > 2):
        res2bin(sys.argv[1], sys.argv[2])
    else:
        print("Not enough arguments, expected: hdf5 filename, output filename")
        exit()
