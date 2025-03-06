#!/bin/bash

# This script is to check correctness of converted circle model by
# executing source onnx with onnx-runtime and target circle
# with circle-interpreter and compare output results.
#
# HOW TO USE
#
# ./run_value_test.sh <path/to/venv_dir> <model> <rtol atol>
# venv_dir : python virtual environment home directory
# model    : model base name
# rtol atol: (optional), refer numpy.isclose()

set -e

TEST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PATH="$1"; shift
MODEL_NAME="$1"; shift
RTOL_VALUE="${1:-1e-04}"
ATOL_VALUE="${2:-1e-04}"

EXEC_ONNX_SCRIPT=${TEST_PATH}/exec_onnx.py
EXEC_CIRCLE_SCRIPT=${TEST_PATH}/exec_circle.py

MAKE_CIRCLE_INPUT_SCRIPT=${TEST_PATH}/make_circle_input.py
COMP_ONNX_CIRCLE_SCRIPT=${TEST_PATH}/comp_onnx_circle.py

ONNX_FILE="${MODEL_NAME}.onnx"
CIRCLE_FILE="${MODEL_NAME}.circle"

# Execute ONNX, Circle and compare
echo "======================================================================"
if [[ -f ${VENV_PATH}/bin/activate ]]; then
  echo "Enter VENV ${VENV_PATH}"
  source ${VENV_PATH}/bin/activate
fi

# Execute ONNX model and generate input/output files
echo "Run ${EXEC_ONNX_SCRIPT} ${ONNX_FILE}"
python3 ${EXEC_ONNX_SCRIPT} ${ONNX_FILE}

# Convert input/output H5 files to binary file for luci_eval_driver
echo "Run ${MAKE_CIRCLE_INPUT_SCRIPT} ${ONNX_FILE} ${CIRCLE_FILE}"
python3 ${MAKE_CIRCLE_INPUT_SCRIPT} ${ONNX_FILE} ${CIRCLE_FILE}

# Execute Circle model and generate output files
echo "Run ${EXEC_CIRCLE_SCRIPT} ${CIRCLE_FILE}"
python3 ${EXEC_CIRCLE_SCRIPT} ${CIRCLE_FILE}

echo "Run ${COMP_ONNX_CIRCLE_SCRIPT} ${ONNX_FILE} ${CIRCLE_FILE} ${RTOL_VALUE} ${ATOL_VALUE}"
python3 ${COMP_ONNX_CIRCLE_SCRIPT} ${ONNX_FILE} ${CIRCLE_FILE} ${RTOL_VALUE} ${ATOL_VALUE}
COMP_RESULT=$?

if [[ -f ${VENV_PATH}/bin/activate ]]; then
  deactivate
fi

exit ${COMP_RESULT}
