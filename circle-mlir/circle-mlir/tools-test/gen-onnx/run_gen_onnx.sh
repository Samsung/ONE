#!/bin/bash

# This script executes run_gen_onnx.py file to generate ONNX model
#
# HOW TO USE
#
# ./run_gen_onnx.sh <path/to/venv_dir> <path/to/models> <model_name> <onnx_name>
# venv_dir   : python virtual environment home directory
# models     : path where python modules exist
# model_name : name of model
# onnx_name  : name of onnx file

THIS_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT_PATH="${THIS_SCRIPT_PATH}/run_gen_onnx.py"

VENV_PATH="$1"; shift
MODELS_ROOT_PATH="$1"; shift
MODEL_NAME="$1"; shift
ONNX_NAME="$1"; shift

PASSED_TAG="${ONNX_NAME}.passed"
GENERATE_LOG="${ONNX_NAME}.log"
rm -f "${PASSED_TAG}"

cat > "${GENERATE_LOG}" <(
  exec 2>&1
  set -ex

  # NOTE enter venv if exist
  if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
  fi

  "python3" "${PY_SCRIPT_PATH}" "${MODELS_ROOT_PATH}" "${MODEL_NAME}" "${ONNX_NAME}"
  if [[ $? -eq 0 ]]; then
    touch "${PASSED_TAG}"
  fi

  if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    deactivate
  fi
)

if [[ ! -f "${PASSED_TAG}" ]]; then
  exit 255
fi
rm -f "${PASSED_TAG}"
exit 0
