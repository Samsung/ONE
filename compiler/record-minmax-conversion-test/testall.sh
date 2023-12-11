#!/bin/bash

# This script tests the basic behavior of record-minmax
#
# HOW TO USE
#
# ./testall.sh <path/to/test.config> <path/to/work_dir> <TEST 1> <TEST 2> ...
# test.config : set ${RECORD_MINMAX_PATH}
# work_dir : build directory of record-minmax-conversion-test (ex: build/compiler/record-minmax-conversion-test)

GEN_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT_PATH="${GEN_SOURCE_PATH}/gen_h5_random_inputs_all.py"
CONFIG_PATH="$1"; shift
BIN_PATH=$(dirname "$CONFIG_PATH")
WORKDIR="$1"; shift
VIRTUALENV="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found RECORD-MINMAX: ${RECORD_MINMAX_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"

# Generate h5 input data
source "${VIRTUALENV}/bin/activate"
  "${VIRTUALENV}/bin/python" "${GEN_SCRIPT_PATH}" \
  --num_data 3 \
  --output_dir "${BIN_PATH}" \
  --artifact_dir ${WORKDIR} \
  --model "$@"

if [[ $? -ne 0 ]]; then
  echo "FAILED TO GENERATE INPUT"
  exit 255
fi

for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"

  PASSED_TAG="${BIN_PATH}/${TESTCASE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${BIN_PATH}/${TESTCASE}.log" <(
    exec 2>&1
    set -ex

    # Run record-minmax
    "${RECORD_MINMAX_PATH}" \
      --input_model "${TESTCASE_FILE}.circle" \
      --input_data "${BIN_PATH}/${TESTCASE}.tflite.input.h5" \
      --output_model "${BIN_PATH}/${TESTCASE}.out.circle"

    if [[ $? -ne 0 ]]; then
      echo "FAILED TO GENERATE CIRCLE OUTPUT"
      continue
    fi

    # Run record-minmax with auto generated random input
    "${RECORD_MINMAX_PATH}" \
      --input_model "${TESTCASE_FILE}.circle" \
      --output_model "${BIN_PATH}/${TESTCASE}.outr.circle"

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
