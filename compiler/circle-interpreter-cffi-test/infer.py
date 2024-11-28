import argparse
import h5py
import numpy as np
from pathlib import Path
import re
import sys

############ Managing paths for the artifacts required by the test.


def extract_test_args(s):
    p = re.compile('eval\\((.*)\\)')
    result = p.search(s)
    return result.group(1)


parser = argparse.ArgumentParser()
parser.add_argument('--lib_path', type=str, required=True)
parser.add_argument('--test_list', type=str, required=True)
parser.add_argument('--artifact_dir', type=str, required=True)
args = parser.parse_args()

with open(args.test_list) as f:
    contents = [line.rstrip() for line in f]
# remove newline and comments.
eval_lines = [line for line in contents if line.startswith('eval(')]
test_args = [extract_test_args(line) for line in eval_lines]
test_models = [Path(args.artifact_dir) / f'{arg}.circle' for arg in test_args]
input_data = [
    Path(args.artifact_dir) / f'{arg}.opt/metadata/tc/input.h5' for arg in test_args
]
expected_output_data = [
    Path(args.artifact_dir) / f'{arg}.opt/metadata/tc/expected.h5' for arg in test_args
]

############ CFFI test

from cffi import FFI

ffi = FFI()
ffi.cdef("""
  typedef struct InterpreterWrapper InterpreterWrapper;

  const char *get_last_error(void);
  void clear_last_error(void);
  InterpreterWrapper *Interpreter_new(const uint8_t *data, const size_t data_size);
  void Interpreter_delete(InterpreterWrapper *intp);
  void Interpreter_interpret(InterpreterWrapper *intp);
  void Interpreter_writeInputTensor(InterpreterWrapper *intp, const int input_idx, const void *data, size_t input_size);
  void Interpreter_readOutputTensor(InterpreterWrapper *intp, const int output_idx, void *output, size_t output_size);
""")
C = ffi.dlopen(args.lib_path)


def check_for_errors():
    error_message = ffi.string(C.get_last_error()).decode('utf-8')
    if error_message:
        C.clear_last_error()
        raise RuntimeError(f'C++ Exception: {error_message}')


def error_checked(func):
    """
    Decorator to wrap functions with error checking.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        check_for_errors()
        return result

    return wrapper


Interpreter_new = error_checked(C.Interpreter_new)
Interpreter_delete = error_checked(C.Interpreter_delete)
Interpreter_interpret = error_checked(C.Interpreter_interpret)
Interpreter_writeInputTensor = error_checked(C.Interpreter_writeInputTensor)
Interpreter_readOutputTensor = error_checked(C.Interpreter_readOutputTensor)

for idx, model_path in enumerate(test_models):
    with open(model_path, "rb") as f:
        model_data = ffi.from_buffer(bytearray(f.read()))

    try:
        intp = Interpreter_new(model_data, len(model_data))

        # Set inputs
        h5 = h5py.File(input_data[idx])
        input_values = h5.get('value')
        input_num = len(input_values)
        for input_idx in range(input_num):
            arr = np.array(input_values.get(str(input_idx)))
            c_arr = ffi.from_buffer(arr)
            Interpreter_writeInputTensor(intp, input_idx, c_arr, arr.nbytes)
        # Do inference
        Interpreter_interpret(intp)
        # Check outputs
        h5 = h5py.File(expected_output_data[idx])
        output_values = h5.get('value')
        output_num = len(output_values)
        for output_idx in range(output_num):
            arr = np.array(output_values.get(str(output_idx)))
            result = np.empty(arr.shape, dtype=arr.dtype)
            Interpreter_readOutputTensor(intp, output_idx, ffi.from_buffer(result),
                                         arr.nbytes)
            if not np.allclose(result, arr):
                raise RuntimeError("Wrong outputs")

        Interpreter_delete(intp)
    except RuntimeError as e:
        print(e)
        sys.exit(-1)
