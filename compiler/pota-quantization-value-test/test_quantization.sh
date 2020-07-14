#!/bin/bash

# This script tests the basic behavior of record-minmax
#
# HOW TO USE
#
# ./test_quantization.sh <path/to/test.config> <path/to/work_dir> <TEST 1> <TEST 2> ...
# test.config : set ${RECORD_MINMAX_PATH} and ${CIRCLE_QUANTIZER_PATH}
# work_dir : build directory of quantization-value-test (ex: build/compiler/quantization-value-test)

SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT_PATH="${SOURCE_PATH}/compare_tensors.py"
CONFIG_PATH="$1"; shift
TEST_INPUT_PATH="${SOURCE_PATH}/test_inputs"
WORKDIR="$1"; shift
VIRTUALENV="${WORKDIR}/venv"

source "${CONFIG_PATH}"

echo "-- Found CIRCLE_QUANTIZER: ${CIRCLE_QUANTIZER_PATH}"
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

  PASSED_TAG="${TESTCASE_FILE}.quantization.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TESTCASE_FILE}_quantization.log" <(
    exec 2>&1
    set -ex

    # Run circle-quantizer with --quantize_with_minmax
    "${CIRCLE_QUANTIZER_PATH}" \
      --quantize_with_minmax float32 "${DTYPE}" "${GRANULARITY}" \
      "${TESTCASE_FILE}.minmax_recorded.circle" \
      "${TESTCASE_FILE}.quantized.circle" 

    # Dump scale, zp, weights values (circle-tensordump)
    "${CIRCLE_TENSORDUMP_PATH}" \
      "${TESTCASE_FILE}.quantized.circle" \
      --tensors_to_hdf5 "${TESTCASE_FILE}.quantized.circle.h5"

    # Compare result
    "${VIRTUALENV}/bin/python" "${COMPARE_SCRIPT_PATH}" \
      --input_h5 "${TESTCASE_FILE}.quantized.circle.h5" \
      --expect_dir "${SOURCE_PATH}/expected_outputs/${MODELNAME}/${GRANULARITY}/${DTYPE}/quantization" \
      --mode quantization

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
