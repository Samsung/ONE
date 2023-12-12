#!/bin/bash

# This script tests the parallel behavior of record-minmax
#
# HOW TO USE
#
# ./test_parallel_record_minmax.sh <path/to/test.config> <path/to/work_dir> <TEST 1> <TEST 2> ...
# test.config : set ${RECORD_MINMAX_PATH} and ${CIRCLE2CIRCLE_PATH}
# work_dir : build directory of quantization-value-test (ex: build/compiler/quantization-value-test)

SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT_PATH="${SOURCE_PATH}/compare_tensors_all.py"
CONFIG_PATH="$1"; shift
BIN_PATH=$(dirname "${CONFIG_PATH}")
TEST_INPUT_PATH="${SOURCE_PATH}/test_inputs"
GEN_SCRIPT_PATH="${BIN_PATH}/gen_h5_explicit_inputs_all.py"
WORKDIR="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found RECORD-MINMAX: ${RECORD_MINMAX_PATH}"
echo "-- Found CIRCLE_TENSORDUMP: ${CIRCLE_TENSORDUMP_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

TEST_PARAMS="$@"

# Generate h5 input data
source "${VIRTUALENV}/bin/activate"
"${VIRTUALENV}/bin/python" "${GEN_SCRIPT_PATH}" \
  --output_dir ${BIN_PATH} \
  --artifact_dir ${WORKDIR} \
  --input_dir ${TEST_INPUT_PATH} \
  --test_param "$TEST_PARAMS"

if [[ $? -ne 0 ]]; then
  echo "FAILED TO GENERATE INPUT"
  exit 255
fi

pushd "${WORKDIR}"
while [ "$1" != "" ]; do
  MODELNAME=$1; shift
  GRANULARITY=$1; shift
  DTYPE=$1; shift
  TESTCASE="${MODELNAME}.${GRANULARITY}.${DTYPE}"

  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"
  TEST_RESULT_FILE="${BIN_PATH}/${TESTCASE}"

  PASSED_TAG="${TEST_RESULT_FILE}.parallel_record_minmax.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TEST_RESULT_FILE}_parallel_record_minmax.log" <(
    exec 2>&1
    set -ex

    # Run parallel record-minmax
    "${RECORD_MINMAX_PATH}" \
      --input_model "${TEST_RESULT_FILE}.fake_quantized.circle" \
      --input_data "${TEST_RESULT_FILE}.input.h5" \
      --output_model "${TEST_RESULT_FILE}.parallel_minmax_recorded.circle" \
      --num_threads 4
    # Dump min/max values (circle-tensordump)
    "${CIRCLE_TENSORDUMP_PATH}" \
      "${TEST_RESULT_FILE}.parallel_minmax_recorded.circle" \
      --tensors_to_hdf5 "${TEST_RESULT_FILE}.parallel_minmax_recorded.circle.h5"
  )
done
popd

# Compare result
"${VIRTUALENV}/bin/python" "${COMPARE_SCRIPT_PATH}" \
  --test_param "${TEST_PARAMS}" \
  --bin_dir ${BIN_PATH} \
  --source_dir ${SOURCE_PATH} \
  --mode record_minmax

if [[ $? -ne 0 ]]; then
  exit 255
fi

echo "PASSED"
exit 0
