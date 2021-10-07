#!/bin/bash

# This script compares the selected circle model by recipe and by 'circle-opselector'
# And to compare the two circle models, 'luci-value-test' which compare equivalence of tflite and circle model was used.
#
# HOW TO USE
#
# ./value_test.sh <path/to/bin_dir> <path/to/work_dir> <path/to/venv_dir> <path/to/circle-opselector-dir> <TEST 1> <TEST 2> ...
VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/luci_eval_verifier.py"
BINDIR="$1"; shift
ARTIFACTS_DIR="$1"; shift
VIRTUALENV="$1"; shift
INTERPRETER_DRIVER_PATH="$1"; shift
COP_DRIVER_PATH="$1"; shift
rm -rf ${ARTIFACTS_DIR}/cop_tmp 
mkdir ${ARTIFACTS_DIR}/cop_tmp
WORKDIR="${ARTIFACTS_DIR}/cop_tmp"

TESTED=()
PASSED=()
FAILED=()
for TESTCASE in "$@"; do
  SPLITER=(${TESTCASE//|/ })
  CIRCLE_ORIGIN_NAME=${SPLITER[0]}
  CIRCLE_SELECTED_NAME=${SPLITER[1]}
  NODES_SELECTED="${SPLITER[2]}"
  TESTED+=("${CIRCLE_SELECTED_NAME}")
  CIRCLE_ORIGIN="${ARTIFACTS_DIR}/${CIRCLE_ORIGIN_NAME}.circle"
  TFLITE_SELECTED="${ARTIFACTS_DIR}/${CIRCLE_SELECTED_NAME}.tflite"  # created by recipe
  cp "${TFLITE_SELECTED}" "${WORKDIR}"  # copy tflite to workdir
  CIRCLE_SELECTED="${WORKDIR}/${CIRCLE_SELECTED_NAME}.circle"  # created by circle-opselector
  
  # select nodes using circle-opselector and locate the result in work directory
  eval "${COP_DRIVER_PATH} --by_id \"${NODES_SELECTED}\" ${CIRCLE_ORIGIN} ${CIRCLE_SELECTED}"

  # test log
  TESTCASE_FILE="${WORKDIR}/${CIRCLE_SELECTED_NAME}"
  TEST_RESULT_FILE="${BINDIR}/${CIRCLE_SELECTED_NAME}"
  PASSED_TAG="${TEST_RESULT_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}.log" <(
    exec 2>&1
    set -ex
    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${VERIFY_SCRIPT_PATH}" \
    --driver "${INTERPRETER_DRIVER_PATH}" \
    --model "${TESTCASE_FILE}"
    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
  )

  if [[ -f "${PASSED_TAG}" ]]; then
    PASSED+=("${TESTCASE}")
  else
    FAILED+=("${TESTCASE}")
  fi
done
if [[ ${#TESTED[@]} -ne ${#PASSED[@]} ]]; then
  echo "FAILED"
  for TEST in "${FAILED[@]}"
  do
    echo "- ${TEST}"
  done
  exit 255
fi

echo "PASSED"
exit 0
