#!/bin/bash

# This script tests the basic behavior of weights-only quantization
#
# HOW TO USE
#
# ./test_quantization.sh <path/to/test.config> <path/to/work_dir> <TEST 1> <TEST 2> ...
# test.config : set ${CIRCLE_QUANTIZER_PATH} and ${CIRCLE_TENSORDUMP_PATH}"
# work_dir : build directory of quantization-value-test (ex: build/compiler/quantization-value-test)

SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT_PATH="${SOURCE_PATH}/compare_tensors_all.py"
CONFIG_PATH="$1"; shift
BIN_PATH=$(dirname "${CONFIG_PATH}")
TEST_INPUT_PATH="${SOURCE_PATH}/test_inputs"
WORKDIR="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found CIRCLE_QUANTIZER: ${CIRCLE_QUANTIZER_PATH}"
echo "-- Found CIRCLE_TENSORDUMP: ${CIRCLE_TENSORDUMP_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

TEST_PARAMS="$@"

pushd "${WORKDIR}"
while [ "$1" != "" ]; do
  MODELNAME=$1; shift
  GRANULARITY=$1; shift
  DTYPE=$1; shift
  TESTCASE="${MODELNAME}.${GRANULARITY}.${DTYPE}"

  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"
  TEST_RESULT_FILE="${BIN_PATH}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.wo_quantization.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}_wo_quantization.log" <(
    exec 2>&1
    set -ex

    # Run circle-quantizer with --quantize_weights
    "${CIRCLE_QUANTIZER_PATH}" \
      --quantize_weights float32 "${DTYPE}" "${GRANULARITY}" \
      "${WORKDIR}/${MODELNAME}.circle" \
      "${TEST_RESULT_FILE}.wo_quantized.circle"

    # Dump scale, zp, weights values (circle-tensordump)
    "${CIRCLE_TENSORDUMP_PATH}" \
      "${TEST_RESULT_FILE}.wo_quantized.circle" \
      --tensors_to_hdf5 "${TEST_RESULT_FILE}.wo_quantized.circle.h5"
  )
done
popd

# Compare result
"${VIRTUALENV}/bin/python" "${COMPARE_SCRIPT_PATH}" \
  --test_param "${TEST_PARAMS}" \
  --bin_dir ${BIN_PATH} \
  --source_dir ${SOURCE_PATH} \
  --mode weights_only_quantization

if [[ $? -ne 0 ]]; then
  exit 255
fi

echo "PASSED"
exit 0
