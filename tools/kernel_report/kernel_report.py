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

import os
import argparse


class Backend:
    def __init__(self):
        self.backends = {}
        self.backends["acl_cl"] = False
        self.backends["acl_neon"] = False
        self.backends["cpu"] = False
        self.backends["srcn"] = False


class KernelReporter(object):
    def __init__(self, args):
        # TODO: Remove os defendency - '/'
        if args.base[0] != '/':
            self.onertBase = os.getcwd() + '/' + args.base
        else:
            self.onertBase = args.base
        self.opListFile = "core/include/ir/Operations.lst"
        self.operations = []
        self.kernelGeneratorFile = "KernelGenerator.h"
        self.kernelMap = {}

    def parseOpList(self):
        #buf = open(self.onertBase + '/' + self.opListFile, "r")

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
            self.kernelMap[op] = Backend()

        backendLists = ["acl_cl", "acl_neon", "cpu", "srcn"]

        for backend in backendLists:
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
        backendLists = ["acl_cl", "acl_neon", "cpu", "srcn"]
        line = ""
        for backend in backendLists:
            line = line + "{0:^9}".format(backend)
        print('{0:30}{1}'.format("", line))

        counts = [0, 0, 0, 0]

        for op in self.operations:
            line = ""
            for i in range(0, 4, 1):
                support = self.kernelMap[op].backends[backendLists[i]]
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
        backendLists = ["acl_cl", "acl_neon", "cpu", "srcn"]
        line = ""
        for backend in backendLists:
            line = line + "|" + backend
        print("|" + line)
        print("-|-|-|-|-")

        counts = [0, 0, 0, 0]

        for op in self.operations:
            line = ""
            for i in range(0, 4, 1):
                support = self.kernelMap[op].backends[backendLists[i]]
                if support:
                    line = line + "|" + "O"
                    counts[i] += 1
                else:
                    line = line + "|" + "-"
            print(op + line)

        line = ""
        for count in counts:
            line = line + "|" + str(count)

        print("-|-|-|-|-")
        print("TOTAL COUNT" + line)

    def run(self):
        self.parseOpList()
        self.generateKernelMap()
        self.printResult()

        self.printMDFormat()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("base", type=str, help="onert base directory")
    args = arg_parser.parse_args()

    report = KernelReporter(args)
    report.run()
