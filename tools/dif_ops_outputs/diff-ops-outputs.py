# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import subprocess


class OneInfo:
    def __init__(self, filename):
        self.subg_id = -1
        self.op_id = -1
        self.count = -1
        self.output_id = -1

        self.filename = filename

        self._parse()
        assert (self.subg_id != -1)
        assert (self.op_id != -1)
        assert (self.count != -1)
        assert (self.output_id != -1)

    def _parse(self):
        # format of filename is
        #   subg-4_op-2_99_o-0_opseq-544.h5
        toks = self.filename.split("_")
        for i in range(0, 4):
            if i == 0:
                subg_toks = toks[i].split("-")
                assert (subg_toks[0] == "subg")
                self.subg_id = subg_toks[1]
            if i == 1:
                op_toks = toks[i].split("-")
                assert (op_toks[0] == "op")
                self.op_id = op_toks[1]
            if i == 2:
                count_tok = toks[i]
                self.count = count_tok
            if i == 3:
                output_toks = toks[i].split("-")
                assert (output_toks[0] == "o")
                self.output_id = output_toks[1]


class ComparingInfo:
    def __init__(self, one_info):
        # format of filename is
        #   subg-4_op-2_99_o-0.h5 or
        sep = "_"
        self.filename = self._chunk2("subg", one_info.subg_id) + sep + \
                        self._chunk2("op", one_info.op_id) + sep + \
                        self._chunk1(one_info.count) + sep + \
                        self._chunk2("o", one_info.output_id) + \
                        ".h5"

    def _chunk2(self, key, id):
        return key + "-" + id

    def _chunk1(self, id):
        return id


class Diff:
    def __init__(self, dir1, dir2, delta):
        def appendSeparator(dir_path):
            if (dir_path[len(dir_path) - 1] == '/'):
                return dir_path
            else:
                return dir_path + "/"

        self._dir1 = appendSeparator(dir1)
        self._dir2 = appendSeparator(dir2)
        self._delta = delta

        assert (os.path.isdir(self._dir1))
        assert (os.path.isdir(self._dir2))

        self.finish = False
        self.exit_code = 0

    def run(self):
        seq_file = self._dir1 + "sequence.txt"
        assert (os.path.isfile(seq_file))

        with open(seq_file) as f:
            for line in f:
                line = line.strip()
                print(line + " from sequence.txt")
                one_info = OneInfo(line)
                comp_info = ComparingInfo(one_info)

                path1 = self._dir1 + one_info.filename
                path2 = self._dir2 + comp_info.filename

                assert (os.path.isfile(path1))
                if (not os.path.isfile(path2)):
                    print("Cannot find comparing file: " + path2 + " -> skip")
                    continue

                # h5 diff
                if (self._compare(path1, path2) == False):
                    # print("\n\nValues are different: please run the following for details:")
                    # print("h5diff -d 0.001 -v " + path1 + " " + path2)
                    # let's finish
                    self.exit_code = 1
                    # return

    def _compare(self, fullpath1, fullpath2):
        def showDiffErrorMsg(cmd):
            print(
                "============================================================================"
            )

            os.system(cmd)  # run once to show error message

            print("\n The above is the result of the following command: ")
            print("\n$ " + cmd)

            print(
                "============================================================================"
            )
            print("\n\nplease press [Enter] to continue")
            input()  # pause

        try:
            cmd = "h5diff -d " + str(self._delta) + " -v " + fullpath1 + " " + fullpath2
            output = subprocess.check_output(cmd, shell=True)
            if ("not comparable" in str(output)):  # type of output is Bytes
                # TODO find out how to pretty print 'output'
                showDiffErrorMsg(cmd)
                return False
            else:
                return True

        except subprocess.CalledProcessError as grepexc:
            showDiffErrorMsg(cmd)
            # print("error code", grepexc.returncode)
            # print(str(grepexc.output)) # This does not handle CR well
            return False


if __name__ == '__main__':
    # Define argument and read
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-1",
        "--one",
        type=str,
        required=True,
        help="dir containng op outputs by ONE runtime")
    arg_parser.add_argument(
        "-2",
        "--comparing",
        type=str,
        required=True,
        help="dir containng op outputs by other runtime")
    arg_parser.add_argument("-d", "--delta", default="0.001")

    args = arg_parser.parse_args()

    # Call main function
    diff = Diff(args.one, args.comparing, args.delta)
    diff.run()

    exit(diff.exit_code)
