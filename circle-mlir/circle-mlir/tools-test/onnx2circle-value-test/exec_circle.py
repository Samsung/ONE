#!/usr/bin/env python3

import os
import sys
import subprocess

from pathlib import Path


def exec_model(circle_model):
    # Execute circle-interpreter
    one_compiler_root = os.getenv('ONE_COMPILER_ROOT')
    if not one_compiler_root:
        one_compiler_root = '/usr/share/one'
    driver = one_compiler_root + '/bin/circle-interpreter'
    input_data = circle_model + '.input'
    output_res_data = circle_model + '.output'
    return subprocess.run([driver, circle_model, input_data, output_res_data], check=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        filepath = Path(sys.argv[0])
        sys.exit('Usage: ' + filepath.name + ' [model.cirle]')

    exec_model(sys.argv[1])
