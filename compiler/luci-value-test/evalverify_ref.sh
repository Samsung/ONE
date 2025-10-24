#!/bin/bash

# This script verifies the basic behavior of luci interpreter
#
# HOW TO USE
#
# ./evalverify_ref.sh <path/to/bin_dir> <path/to/ref_dir> <path/to/eval_driver> \
#                     <TEST 1> <TEST 2> ...
# bin_dir  : build directory of luci-value-test (ex: build/compiler/luci-value-test)
# ref_dir  : artifacts directoy where reference test materials exist
# eval_driver : luci_eval_driver path for evaluation

VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/luci_eval_verifier_ref.py"
BINDIR="$1"; shift
REFDIR="$1"; shift
INTERPRETER_DRIVER_PATH="$1"; shift

TESTED=()
PASSED=()
FAILED=()

for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${REFDIR}/${TESTCASE}"
  TEST_RESULT_FILE="${BINDIR}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}.log" <(
    exec 2>&1
    set -ex

    "python3" "${VERIFY_SCRIPT_PATH}" \
    --driver "${INTERPRETER_DRIVER_PATH}" \
    --model_ref "${TESTCASE_FILE}" \
    --work_path "${TEST_RESULT_FILE}"

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
