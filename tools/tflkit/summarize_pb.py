import argparse
import os
import subprocess
import re


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def CheckExt(choices, name):
    ext = os.path.splitext(name)[1][1:]
    if ext not in choices:
        parser.error("file does not end with one of {}".format(choices))
    return name


def PrintName(path):
    print("")
    print('\t' + os.path.basename(path))
    print("")


def PrintInput(data):
    print("Inputs")
    sub = data.split('(', 1)[1].rsplit(')', 1)[0]
    for i in re.split(', ', sub):
        print('\t' + i)


def PrintOutput(data):
    print("Outputs")
    sub = re.findall(r'\((.*?)\)', data)
    for i in sub:
        print('\t' + i)


def PrintOpType(data):
    print("Op Types")
    cnt = 0
    sub = data.rsplit(':', 1)[1].split(',')
    for i in sub:
        cnt = cnt + 1
        print('\t' + i.lstrip())
    print('\t{0} Total'.format(cnt))


def BuildTensorFlowSummarizeGraph(tensorflow_path):
    with cd(tensorflow_path):
        subprocess.call(
            ['bazel', 'build', 'tensorflow/tools/graph_transforms:summarize_graph'])


def SummarizeGraph(args):
    if args.verbose is True:
        vstr = ""
    PrintName(args.input_file)
    with cd(args.tensorflow_path):
        proc = subprocess.Popen([
            'bazel-bin/tensorflow/tools/graph_transforms/summarize_graph',
            '--in_graph=' + args.input_file
        ],
                                stdout=subprocess.PIPE)
        while True:
            line = proc.stdout.readline().decode()
            if args.verbose:
                vstr += line
            if line != '':
                if 'inputs:' in line:
                    PrintInput(line)
                elif 'outputs:' in line:
                    PrintOutput(line)
                elif 'Op types used:' in line:
                    PrintOpType(line)
            else:
                break

    if args.verbose:
        print(vstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        required=True,
                        type=lambda s: CheckExt((['pb']), s),
                        help='pb file to read')
    parser.add_argument('--tensorflow_path',
                        default='../../externals/tensorflow',
                        help='TensorFlow git repository path')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Build summarize_graph in external/tensorflow
    BuildTensorFlowSummarizeGraph(args.tensorflow_path)

    # Summarize graph
    SummarizeGraph(args)
