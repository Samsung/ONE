#!/bin/bash

# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script tests the basic behavior of dalgona
#
# HOW TO USE
#
# ./test_single_op.sh <path/to/test.config> <path/to/work_dir> <path/to/venv> <TEST 1> <TEST 2> ...
# test.config : set ${DALGONA_PATH}
# work_dir : archive of common-artifacts (ex: build/compiler/common-artifacts)
# venv : virtual environment for python execution

CONFIG_PATH="$1"; shift
BIN_PATH=$(dirname "$CONFIG_PATH")
GEN_SCRIPT_PATH="${BIN_PATH}/GenH5RandomInputs.py"
TEST_SCRIPT_PATH="${BIN_PATH}/SingleOperatorTest.py"
WORKDIR="$1"; shift
VIRTUALENV="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found DALGONA: ${DALGONA_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"
for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${WORKDIR}/${TESTCASE}"

  PASSED_TAG="${BIN_PATH}/${TESTCASE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${BIN_PATH}/${TESTCASE}.log" <(
    exec 2>&1
    set -ex

    # Generate random h5 input data
    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${GEN_SCRIPT_PATH}" \
    --model "${TESTCASE_FILE}.circle" \
    --num_data 3 \
    --output "${BIN_PATH}/${TESTCASE}.circle.input.h5"
    if [[ $? -ne 0 ]]; then
      echo "FAILED TO GENERATE INPUT"
      continue
    fi

    # Run dalgona with test script(SingleOperatorTest.py)
    "${DALGONA_PATH}" \
      --input_model "${TESTCASE_FILE}.circle" \
      --input_data "${BIN_PATH}/${TESTCASE}.circle.input.h5" \
      --analysis "${TEST_SCRIPT_PATH}" \
      --analysis_args "${TESTCASE_FILE}.circle"

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
