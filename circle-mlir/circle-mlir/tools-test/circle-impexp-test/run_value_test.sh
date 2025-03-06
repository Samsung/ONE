#!/bin/bash

# This script is to check correctness of import module by
# executing source circle and target circle, from circle-impexp,
# with circle-interpreter and compare output results.
#
# HOW TO USE
#
# ./run_value_test.sh <path/to/venv_dir> <model>
# venv_dir : python virtual environment home directory
# model    : model base name

set -e

TEST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PATH="$1"; shift
MODEL_NAME="$1"; shift

EXEC_ONNX_SCRIPT=${TEST_PATH}/exec_onnx.py
EXEC_CIRCLE_SCRIPT=${TEST_PATH}/exec_circle.py

MAKE_CIRCLE_INPUT_SCRIPT=${TEST_PATH}/make_circle_input.py
COMP_CIRCLE_CIRCLE_SCRIPT=${TEST_PATH}/comp_circle_circle.py

# NOTE ONNX_FILE is just to acquire I/O h5 files
ONNX_FILE="${MODEL_NAME}.onnx"
CIRCLE_SOURCE_FILE="${MODEL_NAME}.circle"
CIRCLE_TARGET_FILE="${MODEL_NAME}.2.circle"

# Execute Circle, Circle.2 and compare
echo "======================================================================"
if [[ -f ${VENV_PATH}/bin/activate ]]; then
  echo "Enter VENV ${VENV_PATH}"
  source ${VENV_PATH}/bin/activate
fi

# TODO execute scripts

COMP_RESULT=0

if [[ -f ${VENV_PATH}/bin/activate ]]; then
  deactivate
fi

exit ${COMP_RESULT}
