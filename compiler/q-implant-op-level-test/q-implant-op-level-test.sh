#!/bin/bash

VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/q_implant_op_level_test.py"
BINDIR="$1"; shift
WORKDIR="$1"; shift
VIRTUALENV="$1"; shift
INTERPRETER_DRIVER_PATH="$1"; shift

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
    --model "${TESTCASE}" \
    --driver "${INTERPRETER_DRIVER_PATH}" \
    --output_dir "${BINDIR}" \
    --input_dir "${TESTCASE_FILE}"

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
