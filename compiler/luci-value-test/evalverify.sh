#!/bin/bash

# This script verifies the basic behavior of luci interpreter
#
# HOW TO USE
#
# ./evalverify.sh <path/to/luci_interpreter_tester> <TEST 1> <TEST 2> ...
VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORKDIR="$1"; shift
INTERPRETER_DRIVER_PATH="$1"; shift
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/eval_verifier.py"

TESTED=()
PASSED=()
FAILED=()

for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"

  PASSED_TAG="${TESTCASE_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TESTCASE_FILE}.log" <(
    exec 2>&1
    set -ex

    "${VERIFY_SCRIPT_PATH}" "--driver" "${INTERPRETER_DRIVER_PATH}" "--model" "${TESTCASE_FILE}.tflite"

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
