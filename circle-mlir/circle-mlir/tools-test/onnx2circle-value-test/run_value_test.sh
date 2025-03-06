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

ONNX_FILE="${MODEL_NAME}.onnx"
CIRCLE_FILE="${MODEL_NAME}.circle"

# Execute ONNX, Circle and compare
echo "======================================================================"
if [[ -f ${VENV_PATH}/bin/activate ]]; then
  echo "Enter VENV ${VENV_PATH}"
  source ${VENV_PATH}/bin/activate
fi

# TODO execute and compare

COMP_RESULT=0

if [[ -f ${VENV_PATH}/bin/activate ]]; then
  deactivate
fi

exit ${COMP_RESULT}
