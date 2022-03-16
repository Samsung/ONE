#!/bin/bash

# This script tests fake quantization with config file
#
# HOW TO USE
#
# ./test_fake_wquant_with_config.sh <path/to/test.config> <path/to/work_dir> <TEST 1> <TEST 2> ...
# test.config : set ${RECORD_MINMAX_PATH} and ${CIRCLE_QUANTIZER_PATH}
# work_dir : build directory of quantization-value-test (ex: build/compiler/quantization-value-test)

SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT_PATH="${SOURCE_PATH}/compare_tensors.py"
CONFIG_PATH="$1"; shift
BIN_PATH=$(dirname "${CONFIG_PATH}")
WORKDIR="$1"; shift

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
  TEST_RESULT_FILE="${BIN_PATH}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.fake_quantized.mixed.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}_fake_quantization_with_config.log" <(
    exec 2>&1
    set -ex

    # Run circle-quantizer with --quantize_dequantize_weights
    "${CIRCLE_QUANTIZER_PATH}" \
      --quantize_dequantize_weights float32 "${DTYPE}" "${GRANULARITY}" \
      --config "${SOURCE_PATH}/config_files/${MODELNAME}/${GRANULARITY}/${DTYPE}/qconf.json" \
      "${WORKDIR}/${MODELNAME}.circle" \
      "${TEST_RESULT_FILE}.fake_quantized.mixed.circle" 

    # Dump weights values (circle-tensordump)
    "${CIRCLE_TENSORDUMP_PATH}" \
      "${TEST_RESULT_FILE}.fake_quantized.mixed.circle" \
      --tensors_to_hdf5 "${TEST_RESULT_FILE}.fake_quantized.mixed.circle.h5"

    # Compare result
    "${VIRTUALENV}/bin/python" "${COMPARE_SCRIPT_PATH}" \
      --input_h5 "${TEST_RESULT_FILE}.fake_quantized.mixed.circle.h5" \
      --expect_dir "${SOURCE_PATH}/expected_outputs/${MODELNAME}_config/${GRANULARITY}/${DTYPE}/fake_quantization" \
      --mode fake_quantization

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
