#!/usr/bin/python3

# This utility wraps contents of input file into C char array, so it can be embedded into C/C++ code

import sys

def generate(intput_filename, output_filename, array_name):
  with open(input_filename, 'r') as input_file:
    file_contents = input_file.read()
  with open(output_filename, 'w') as output_file:
    prelude = 'const char *' + array_name + ' = R"magic_9681(' # random number
    finishing = ')magic_9681";\n';
    output_file.write(prelude + file_contents + finishing)

if __name__ == '__main__':
  if len(sys.argv) != 4:
    print('Usage: embed <input file> <output_file> <name of generated array>')
    exit(1)
  input_filename = sys.argv[1]
  output_filename = sys.argv[2]
  array_name = sys.argv[3]
  generate(input_filename, output_filename, array_name)
