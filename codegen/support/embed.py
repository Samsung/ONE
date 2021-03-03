#!/usr/bin/python3

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

# This utility wraps contents of input file into C char array, so it can be embedded into C/C++ code
# it also strips comments to eliminate redundant data

import sys


def generate(intput_filename, output_filename, array_name):
    file_contents = ""
    with open(input_filename, 'r') as input_file:
        line = input_file.readline()
        while line:
            if line.find("//") != 0:
                file_contents += line
                file_contents += "\n"
            line = input_file.readline()
    with open(output_filename, 'w') as output_file:
        prelude = 'const char *' + array_name + ' = R"magic_9681('  # random number
        finishing = ')magic_9681";\n'
        output_file.write(prelude + file_contents + finishing)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: embed <input file> <output_file> <name of generated array>')
        exit(1)
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    array_name = sys.argv[3]
    generate(input_filename, output_filename, array_name)
