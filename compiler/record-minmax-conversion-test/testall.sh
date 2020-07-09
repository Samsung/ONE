#!/bin/bash

# This script tests the basic behavior of record-minmax
#
# HOW TO USE
#
# ./testall.sh <path/to/test.config> <path/to/work_dir> <TEST 1> <TEST 2> ...
# test.config : set ${RECORD_MINMAX_PATH}
# work_dir : build directory of record-minmax-conversion-test (ex: build/compiler/record-minmax-conversion-test)

GEN_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT_PATH="${GEN_SOURCE_PATH}/gen_h5_random_inputs.py"
CONFIG_PATH="$1"; shift
WORKDIR="$1"; shift
VIRTUALENV="${WORKDIR}/venv"

source "${CONFIG_PATH}"

echo "-- Found RECORD-MINMAX: ${RECORD_MINMAX_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"
for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"

  PASSED_TAG="${TESTCASE_FILE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TESTCASE_FILE}.log" <(
    exec 2>&1
    set -ex

    # Generate h5 input data
    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${GEN_SCRIPT_PATH}" \
    --model "${TESTCASE_FILE}.tflite" \
    --num_data 3

    if [[ $? -ne 0 ]]; then
      echo "FAILED TO GENERATE INPUT"
      continue
    fi

    # Run record-minmax
    "${RECORD_MINMAX_PATH}" \
      --input_model "${TESTCASE_FILE}.circle" \
      --input_data "${TESTCASE_FILE}.tflite.input.h5" \
      --output_model "${TESTCASE_FILE}.out.circle" 

    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
  )

  if [[ -f "${PASSED_TAG}" ]]; then
    PASSED+=("$TESTCASE")
  else
    FAILED+=("$TESTCASE")
  fi
done
popd

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
