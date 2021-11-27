#!/bin/bash

# This script verifies that imported without constants copying models executes well in luci_interpreter
#
# HOW TO USE
#
# ./evalverify.sh <path/to/bin_dir> <path/to/work_dir> <TEST 1> <TEST 2> ...
# bin_dir  : build directory of luci-value-test (ex: build/compiler/luci-value-test)
# work_dir : artifacts directory where test materials exist

BINDIR="$1"; shift
WORKDIR="$1"; shift
TEST_DRIVER_PATH="${BINDIR}/test_driver"
TEST_RESULT_DIR="${BINDIR}/result"

TESTED=()
PASSED=()
FAILED=()

mkdir -p "${TEST_RESULT_DIR}"
for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"
  TEST_RESULT_FILE="${TEST_RESULT_DIR}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}.log" <(
    exec 2>&1
    set -ex

    "${TEST_DRIVER_PATH}" --model "${TESTCASE_FILE}.circle"

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
