#!/bin/bash

# This script verifies the tflite and circle execution result values
#
# HOW TO USE
#
# ./eval_driver.sh <path/to/bin_dir> <path/to/work_dir> <path/to/venv_dir> <path/to/intp_dir>
#                  <TEST 1> <TEST 2> ...
# bin_dir  : build directory of luci-pass-value-test (ex: build/compiler/luci-pass-value-test)
# work_dir : artifacts directoy where test materials exist
# venv_dir : python virtual environment home directory
# intp_dir : path to luci_eval_driver from luci-eval-driver

VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/eval_result_verifier.py"
BINDIR="$1"; shift
WORKDIR="$1"; shift
VIRTUALENV="$1"; shift
INTERPRETER_DRIVER_PATH="$1"; shift

TESTED=()
PASSED=()
FAILED=()

for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_TFLITE_FILE="${WORKDIR}/${TESTCASE}.tflite"
  TESTCASE_CIRCLE_FILE="${BINDIR}/${TESTCASE}.pass.circle"
  TEST_RESULT_FILE="${BINDIR}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}.log" <(
    exec 2>&1
    set -ex

    source "${VIRTUALENV}/bin/activate"

    "${VIRTUALENV}/bin/python" "${VERIFY_SCRIPT_PATH}" \
    --driver "${INTERPRETER_DRIVER_PATH}" \
    --tflite "${TESTCASE_TFLITE_FILE}" \
    --circle "${TESTCASE_CIRCLE_FILE}"

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
