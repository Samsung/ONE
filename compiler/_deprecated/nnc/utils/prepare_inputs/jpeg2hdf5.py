from PIL import Image
import numpy as np
import h5py
import sys
import glob
import subprocess
import struct
import datetime

# Generates hdf5 files (and optionally binary files) from JPEGs
# -f - specifies framework to generate them for
# -t - specifies testcases directory (see it's structure in readme)
# -i - specifies input node name of the model that will use them (required by nnkit)
# -r - if files already exist, rewrites them
# -b - enable binary file generation
# -p - allow some sort of parallelism by processing only a subset of files,
#      you need to specify number of processes and run as much of them
#      manually with diferent numbers
#
# Example:
# python3 conv.py -f tfl -t inc_slim/testcases -i input -p 16 1
#

helpstr = 'Usage: -f (tfl | caf) ' + \
                 '-t <testcases_directory> ' + \
                 '[-i <input_layer_name>] ' + \
                 '[-r] [-b]' + \
                 '[-p <number_of_processes> <process number>]'

supported_frameworks = ['tfl', 'caf']
args = {}
# Defaults
args['-p'] = (1, 1)
args['-r'] = False
args['-b'] = False

argc = len(sys.argv)
for i in range(len(sys.argv)):
    arg = sys.argv[i]
    if arg == '-r' or arg == '-b':
        args[arg] = True
    elif arg == '-f' or arg == '-t' or arg == '-i':
        if i + 1 >= argc or sys.argv[i + 1][0] == '-':
            print(arg, " is missing it's value")
            print(helpstr)
            exit()
        args[arg] = sys.argv[i + 1]
    elif arg == '-p':
        if i + 2 >= argc or sys.argv[i + 1][0] == '-' or sys.argv[i + 2][0] == '-':
            print(arg, " is missing some of it's values")
            print(helpstr)
            exit()
        args[arg] = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    elif arg[0] == '-':
        print('Unsupported argument: ', arg)
        exit()

if not ('-f' in args and '-t' in args):
    print('Some arguments are not provided')
    print(helpstr)
    exit()

fw = args['-f']
if not fw in supported_frameworks:
    print('Unsupported framework: ', fw)
    exit()

indirname = args['-t']

if not '-i' in args:
    if fw == 'caf':
        inputname = 'data'
    elif fw == 'tfl':
        inputname = 'input'
else:
    inputname = args['-i']

nproc, proc_num = args['-p']
remove_existing = args['-r']
gen_binary = args['-b']

print('started at', datetime.datetime.now())
testcases = glob.glob(indirname + '/testcase*/')
testcases.sort()
testcases = testcases[proc_num - 1::nproc]

number = 0
for testcase in testcases:
    try:
        infilename = glob.glob(testcase + 'input/*.JPEG')
        if len(infilename) > 0:
            number += 1
            infilename = infilename[0]
            outfilename = testcase + 'input/' + infilename.split('/')[-1] + '.hdf5'
            binoutfilename = testcase + 'input/' + infilename.split('/')[-1] + '.dat'
            found_hdf = len(glob.glob(outfilename)) != 0
            found_bin = len(glob.glob(binoutfilename)) != 0
            if not found_hdf or (not found_bin and gen_binary) or remove_existing:
                with Image.open(infilename) as im:
                    #TODO: check if order is correct here and in other places
                    h = im.size[0]
                    w = im.size[1]
                    s = im.split()
                if len(s) == 3:
                    r, g, b = s
                else:
                    r = s[0]
                    g = s[0]
                    b = s[0]
                rf = r.convert('F')
                gf = g.convert('F')
                bf = b.convert('F')
                rfb = rf.tobytes()
                gfb = gf.tobytes()
                bfb = bf.tobytes()

                made_hdf = False
                if not found_hdf or remove_existing:
                    if fw == 'tfl':
                        reds = np.fromstring(rfb, count=(h * w), dtype='float32')
                        greens = np.fromstring(gfb, count=(h * w), dtype='float32')
                        blues = np.fromstring(bfb, count=(h * w), dtype='float32')

                        dset_shape = (1, h, w, 3)
                        narr = np.ndarray(shape=(0))
                        mixed_ch = []
                        for i in range(h * w):
                            mixed_ch += [
                                reds[i] / 255.0, greens[i] / 255.0, blues[i] / 255.0
                            ]
                        narr = np.append(narr, mixed_ch)
                    elif fw == 'caf':
                        dset_shape = (1, 3, h, w)
                        narr = np.fromstring(rfb + gfb + bfb,
                                             count=(3 * h * w),
                                             dtype='float32')
                        for i in range(3 * h * w):
                            narr[i] /= 255.0
                    if remove_existing:
                        subprocess.call(['rm', '-f', outfilename])
                    with h5py.File(outfilename) as f:
                        # nnkit hdf5_import asserts to use IEEE_F32BE, which is >f4 in numpy
                        dset = f.require_dataset(inputname, dset_shape, dtype='>f4')
                        dset[0] = np.reshape(narr, dset_shape)
                    made_hdf = True

                if gen_binary and (not found_bin or remove_existing):
                    if fw == 'tfl' and made_hdf:
                        l = narr.tolist()
                    else:
                        reds = np.fromstring(rfb, count=(h * w), dtype='float32')
                        greens = np.fromstring(gfb, count=(h * w), dtype='float32')
                        blues = np.fromstring(bfb, count=(h * w), dtype='float32')
                        l = np.ndarray(shape=(0))
                        mixed_ch = []
                        for i in range(h * w):
                            mixed_ch += [
                                reds[i] / 255.0, greens[i] / 255.0, blues[i] / 255.0
                            ]
                        l = np.append(l, mixed_ch)
                        l = l.tolist()
                    with open(binoutfilename, 'wb') as out:
                        out.write(struct.pack('f' * len(l), *l))
                print(number, ': ' + testcase + ' Done')
            else:
                print(testcase, ' nothing to do')
        else:
            print(testcase, ' JPEG not found')
    except:
        print(testcase, " FAILED")
print('started at', ended.datetime.now())
