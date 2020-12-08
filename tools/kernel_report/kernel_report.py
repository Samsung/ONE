#!/usr/bin/python

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
from os.path import dirname, realpath, join


class Backend:
    def __init__(self, backendList):
        self.backends = {}

        for backend in backendList:
            self.backends[backend] = False


class KernelReporter(object):
    def __init__(self, args):
        root_path = dirname(dirname(dirname(realpath(__file__))))
        self.onertBase = join(root_path, "runtime", "onert")
        if args.md5:
            self.printMD5 = True
        else:
            self.printMD5 = False
        self.backendList = args.backends.split(',')
        self.opListFile = "core/include/ir/Operations.lst"
        self.operations = []
        self.kernelGeneratorFile = "KernelGenerator.h"
        self.kernelMap = {}

    def parseOpList(self):
        # Parsing line and get op list
        skipLine = False
        for line in open(self.onertBase + '/' + self.opListFile, "r"):
            # Skip license
            # TODO : Change to skip general comment
            if skipLine:
                if line.startswith(" */"):
                    skipLine = False
                    continue
                continue
            if line.startswith("/*"):
                skipLine = True
                continue

            # Skip comment
            if line.startswith("//"):
                continue

            # Skip macro
            if line.startswith("#"):
                continue

            lineStripped = line.strip()
            if len(lineStripped) == 0:
                continue

            op = lineStripped[3:-1]
            self.operations.append(op)
            self.operations.sort()

    def generateKernelMap(self):
        for op in self.operations:
            self.kernelMap[op] = Backend(self.backendList)

        for backend in self.backendList:
            buf = open(
                self.onertBase + '/backend/' + backend + '/' + self.kernelGeneratorFile,
                "r")

            for line in buf:
                words = line.split()
                if len(words) < 3:
                    continue
                if words[1] != "visit(const":
                    continue

                opName = words[2].split("::")
                if len(opName) < 3:
                    continue

                if opName[2] in self.operations:
                    self.kernelMap[opName[2]].backends[backend] = True

            buf.close()

    def printResult(self):
        print()
        line = ""
        for backend in self.backendList:
            line = line + "{0:^9}".format(backend)
        print('{0:30}{1}'.format("", line))

        counts = []
        for i in range(0, len(self.backendList), 1):
            counts.append(0)

        for op in self.operations:
            line = ""
            for i in range(0, len(self.backendList), 1):
                support = self.kernelMap[op].backends[self.backendList[i]]
                if support:
                    line = line + "{0:^9}".format("O")
                    counts[i] += 1
                else:
                    line = line + "{0:^9}".format("-")
            print('{0:30}{1}'.format(op, line))

        line = ""
        for count in counts:
            line = line + "{0:^9}".format(count)
        print('{0:30}{1}'.format("TOTAL COUNT", line))

    def printMDFormat(self):
        print()
        line = "-"
        for backend in self.backendList:
            line = line + "|" + backend
        print(line)
        line = ""
        for i in range(0, len(self.backendList), 1):
            line = line + "-|"
        print(line + "-")

        counts = []
        for i in range(0, len(self.backendList), 1):
            counts.append(0)

        for op in self.operations:
            line = ""
            for i in range(0, len(self.backendList), 1):
                support = self.kernelMap[op].backends[self.backendList[i]]
                if support:
                    line = line + "|" + "O"
                    counts[i] += 1
                else:
                    line = line + "|" + "-"
            print(op + line)

        line = ""
        for i in range(0, len(self.backendList), 1):
            line = line + "-|"
        print(line + "-")

        line = ""
        for count in counts:
            line = line + "|" + str(count)

        print("TOTAL COUNT" + line)

    def run(self):
        self.parseOpList()
        self.generateKernelMap()

        if self.printMD5:
            self.printMDFormat()
        else:
            self.printResult()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--backends",
        type=str,
        default='cpu,acl_cl,acl_neon',
        help="backend list to report (use comma)")
    arg_parser.add_argument("--md5", action='store_true', help="Print for md5")
    args = arg_parser.parse_args()

    report = KernelReporter(args)
    report.run()
