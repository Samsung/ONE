#!/bin/bash

# HOW TO USE
#
# ./run_circle_ops_test.sh <path/to/venv_dir> <run_path> <models_path> <model_name>
# venv_dir    : python virtual environment home directory
# run_path    : current path where model.ops file exist
# models_path : models source path
# model_name  : model name

set -e

TEST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

VENV_PATH="$1"; shift
RUN_PATH="$1"; shift
MODELS_PATH="$1"; shift
MODEL_NAME="$1"; shift

CHECK_CIRCLE_OPS_SCRIPT=${TEST_PATH}/check_circle_ops.py
CIRCLE_OPS_FILE="${RUN_PATH}/${MODEL_NAME}.circle.ops"

# Execute ONNX, Circle and compare
echo "======================================================================"
if [[ -f ${VENV_PATH}/bin/activate ]]; then
  echo "Enter VENV ${VENV_PATH}"
  source ${VENV_PATH}/bin/activate
fi

# run check_rewrite script
echo "Run ${CHECK_CIRCLE_OPS_SCRIPT} ${MODELS_PATH} ${MODEL_NAME} ${CIRCLE_OPS_FILE}"
python3 ${CHECK_CIRCLE_OPS_SCRIPT} ${MODELS_PATH} ${MODEL_NAME} ${CIRCLE_OPS_FILE}
COMP_RESULT=$?

if [[ -f ${VENV_PATH}/bin/activate ]]; then
  deactivate
fi

exit ${COMP_RESULT}
