#!/bin/bash

# This script verifies the basic behavior of luci interpreter
#
# HOW TO USE
#
# ./evalverify.sh <path/to/bin_dir> <path/to/work_dir> <path/to/venv_dir> <TEST 1> <TEST 2> ...
# bin_dir  : build directory of luci-value-test (ex: build/compiler/luci-value-test)
# work_dir : artifacts directoy where test materials exist
# venv_dir : python virtual environment home directory

VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/luci_eval_verifier.py"
BINDIR="$1"; shift
WORKDIR="$1"; shift
VIRTUALENV="$1"; shift
INTERPRETER_DRIVER_PATH="${BINDIR}/tester/luci_eval_tester"

TESTED=()
PASSED=()
FAILED=()

for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"
  TEST_RESULT_FILE="${BINDIR}/${TESTCASE}"

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
