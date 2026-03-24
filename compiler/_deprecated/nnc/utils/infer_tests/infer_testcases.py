from __future__ import print_function
import sys
import glob
import subprocess
import res2bin
import datetime

# This script uses nnkit to run inference for given model on a given data
# Messages are printed to stderr
# Usage:
# -b - specifies path to nnkit build folder, inside which tools/run is located
# -f - specifies framework ('tfl' for tflite or 'caf' for caffe) that the model belogs to
# -t - specifies path to testcase folder (see it's structure in readme)
# -p - allow some sort of parallelism by processing only a subset of files,
#      you need to specify number of processes and run as much of them
#      manually with diferent numbers
# -r - infer all testcases regardless of whether the result files are present
# last argument(s) is the model to infer
#
# Example of usage:
# python3 infer_testcases.py -f tfl -b /mnt/nncc_ci/nncc_new/build/contrib/nnkit -t /mnt/nncc_ci/images/inc_slim/testcases/ -p 10 1 -r /mnt/nncc_ci/images/inc_slim/models/inception_v3_2018.tflite
#

helpstr = "Expected arguments: -b <path_to_nnkit>" + \
                               "-f (tfl | caf) " + \
                               "-t <testcases_dir> " + \
                               "[-p <nporc> <proc_num>] " + \
                               "[-r] " + \
                               "(<tflite_model_file> | <caffe_prototxt_model> <caffe_caffemodel_file>)"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


nproc = 1
proc_num = 1
min_argc = 8

args = {}
args['-p'] = (1, 1)
args['-r'] = False

argc = len(sys.argv)
for i in range(argc):
    arg = sys.argv[i]
    if arg == '-r':
        args[arg] = True
    elif arg == '-b' or arg == '-f' or arg == '-t':
        if i + 1 >= argc:
            eprint(arg, " is missing it's value")
            eprint(helpstr)
            exit()
        args[arg] = sys.argv[i + 1]
    elif arg == '-p':
        min_argc += 3
        if i + 2 >= argc:
            eprint(arg, " is missing some of it's values")
            eprint(helpstr)
            exit()
        args[arg] = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    elif arg[0] == '-':
        print('Unsupported argument: ', arg)
        exit()

if not ('-b' in args and '-f' in args and '-t' in args):
    eprint('Some arguments are not provided')
    eprint(helpstr)
    exit()

fw = args['-f']
build_path = args['-b']
testcases_dir = args['-t']
nproc, proc_num = args['-p']
remove_existing = args['-r']

if fw == 'tfl':
    model = sys.argv[-1]
    print('Model: ', model)
elif fw == 'caf':
    model_proto = sys.argv[-2]
    model_caffe = sys.argv[-1]
    print('Models: ', model_proto, model_caffe)
else:
    eprint('Unsupported framework:', fw)
    exit()

eprint('started at', datetime.datetime.now())
print('Framework: ', fw)
print('Path to nnkit: ', build_path)
print('Testcases folder: ', testcases_dir)

hdf_suffix = '.hdf5'
bin_suffix = '.dat'


def get_command_caf(infilename, outfilename, proto, caffemodel):
    return [
        build_path + "/tools/run/nnkit-run", "--pre",
        build_path + "/actions/HDF5/libnnkit_HDF5_import_action.so", "--pre-arg",
        infilename, "--backend", build_path + "/backends/caffe/libnnkit_caffe_backend.so",
        "--backend-arg", proto, "--backend-arg", caffemodel, "--post",
        build_path + "/actions/HDF5/libnnkit_HDF5_export_action.so", "--post-arg",
        outfilename
    ]


def get_command_tfl(infilename, outfilename, model_file):
    return [
        build_path + "/tools/run/nnkit-run", "--pre",
        build_path + "/actions/HDF5/libnnkit_HDF5_import_action.so", "--pre-arg",
        infilename, "--backend",
        build_path + "/backends/tflite/libnnkit_tflite_backend.so", "--backend-arg",
        model_file, "--post", build_path + "/actions/builtin/libnnkit_show_action.so",
        "--post", build_path + "/actions/HDF5/libnnkit_HDF5_export_action.so",
        "--post-arg", outfilename
    ]


testcase_num = 0
testcases = glob.glob(testcases_dir + '/testcase*')

#testcases = [t
#             for t in testcases
#             if remove_existing
#             or len(glob.glob(t + '/output/output' + hdf_suffix)) == 0
#             or len(glob.glob(t + '/output/output' + bin_suffix)) == 0]
testcases = testcases[proc_num - 1::nproc]

testcases.sort()
for testcase in testcases:
    testcase_num += 1
    try:
        infile = glob.glob(testcase + '/input/*' + hdf_suffix)
        if len(infile) > 0:
            infile = infile[0]
            outfile = testcase + '/output/output' + hdf_suffix
            outfile_bin = testcase + '/output/output' + bin_suffix
            if len(glob.glob(outfile)) == 0 or remove_existing:
                if fw == 'tfl':
                    command = get_command_tfl(infile, outfile, model)
                elif fw == 'caf':
                    command = get_command_caf(infile, outfile, model_proto, model_caffe)
                #subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.call(command)
            if len(glob.glob(outfile_bin)) == 0 or remove_existing:
                res2bin.res2bin(outfile, outfile_bin)
            eprint(testcase_num, "/", len(testcases))
        else:
            eprint(testcase, ': input not found')
    except:
        eprint(testcase, 'failed')

eprint('ended at', datetime.datetime.now())
