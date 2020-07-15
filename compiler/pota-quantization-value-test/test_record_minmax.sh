#!/bin/bash

# This script tests the basic behavior of record-minmax
#
# HOW TO USE
#
# ./test_record_minmax.sh <path/to/test.config> <path/to/work_dir> <TEST 1> <TEST 2> ...
# test.config : set ${RECORD_MINMAX_PATH} and ${CIRCLE2CIRCLE_PATH}
# work_dir : build directory of quantization-value-test (ex: build/compiler/quantization-value-test)

SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT_PATH="${SOURCE_PATH}/gen_h5_explicit_inputs.py"
COMPARE_SCRIPT_PATH="${SOURCE_PATH}/compare_tensors.py"
CONFIG_PATH="$1"; shift
BIN_PATH=$(dirname "${CONFIG_PATH}")
TEST_INPUT_PATH="${SOURCE_PATH}/test_inputs"
WORKDIR="$1"; shift
VIRTUALENV="${WORKDIR}/venv_1_13_2"

source "${CONFIG_PATH}"

echo "-- Found RECORD-MINMAX: ${RECORD_MINMAX_PATH}"
echo "-- Found CIRCLE_TENSORDUMP: ${CIRCLE_TENSORDUMP_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"
while [ "$1" != "" ]; do  
  MODELNAME=$1; shift
  GRANULARITY=$1; shift
  DTYPE=$1; shift
  TESTCASE="${MODELNAME}.${GRANULARITY}.${DTYPE}"

  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"
  TEST_RESULT_FILE="${BIN_PATH}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.record_minmax.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}_record_minmax.log" <(
    exec 2>&1
    set -ex

    # Generate h5 input data
    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${GEN_SCRIPT_PATH}" \
      --model "${WORKDIR}/${MODELNAME}.tflite" \
      --input "${TEST_INPUT_PATH}/${MODELNAME}/${GRANULARITY}/${DTYPE}" \
      --output "${TESTCASE_FILE}.input.h5"

    if [[ $? -ne 0 ]]; then
      echo "FAILED TO GENERATE INPUT"
      continue
    fi

    # Run record-minmax
    "${RECORD_MINMAX_PATH}" \
      "${TEST_RESULT_FILE}.fake_quantized.circle" \
      "${TEST_RESULT_FILE}.input.h5" \
      "${TEST_RESULT_FILE}.minmax_recorded.circle" 

    # Dump min/max values (circle-tensordump)
    "${CIRCLE_TENSORDUMP_PATH}" \
      "${TEST_RESULT_FILE}.minmax_recorded.circle" \
      --tensors_to_hdf5 "${TEST_RESULT_FILE}.minmax_recorded.circle.h5"

    # Compare result
    "${VIRTUALENV}/bin/python" "${COMPARE_SCRIPT_PATH}" \
      --input_h5 "${TEST_RESULT_FILE}.minmax_recorded.circle.h5" \
      --expect_dir "${SOURCE_PATH}/expected_outputs/${MODELNAME}/${GRANULARITY}/${DTYPE}/record_minmax" \
      --mode record_minmax

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
