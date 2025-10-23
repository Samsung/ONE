#!/bin/bash

# This script verifies the basic behavior of luci frontend
#
# HOW TO USE
#
# ./readverify.sh <path/to/luci_readtester> <TEST 1> <TEST 2> ...
VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

# set LOG enable to execute/test luci/logex codes
export LUCI_LOG=100

WORKDIR="$1"; shift
VERIFY_BINARY_PATH="$1"; shift

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

    "${VERIFY_BINARY_PATH}" "${TESTCASE_FILE}.circle"

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
