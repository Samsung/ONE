#!/bin/bash

# This script tests the parallel behavior of minmax-embedder
#
# HOW TO USE
#
# ./testall.sh <test.config> <TEST_1> <TEST_2> ...
#
# test.config must contains the following variables:
#  - ARTIFACTS_PATH: path to test models
#  - MINMAX_DATA_GEN: path to minmax_data_gen
#  - MINMAX_EMBEDDER: path to minmax_embedder_driver
#  - CIRCLEDUMP: path to circledump
#  - (H5DUMP) it is assumed in PATH

# <TEST_N> is the name of model under ARTIFACTS_PATH

CONFIG_PATH="$1"; shift; source "${CONFIG_PATH}"
WORK_DIR=$(dirname "${CONFIG_PATH}") # For temporary and report outputs

echo "-- Found ARTIFACTS_PATH: ${ARTIFACTS_PATH}"
echo "-- Found MINMAX_DATA_GEN: ${MINMAX_DATA_GEN}"
echo "-- Found MINMAX_EMBEDDER: ${MINMAX_EMBEDDER}"
echo "-- Found CIRCLEDUMP: ${CIRCLEDUMP}"
echo "-- Found CONFIG_PATH: ${CONFIG_PATH}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORK_DIR}"
for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  TESTCASE_FILE="${ARTIFACTS_PATH}/${TESTCASE}"

  PASSED_TAG="${WORK_DIR}/${TESTCASE}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${WORK_DIR}/${TESTCASE}.log" <(
    exec 2>&1
    set -ex

    # Get model input tensor names
    #INPUT_NAMES=( $("${CIRCLEDUMP}" "${TESTCASE_FILE}.circle" | grep -oP '(?<=^I T\()\d+:\d+') )
    INPUT_NAMES=( $("${CIRCLEDUMP}" "${TESTCASE_FILE}.circle" | grep '^I T' | grep -oE '[^ ]+$') )
    declare -p INPUT_NAMES
    if [[ $? -ne 0 ]]; then
      echo "FAILED TO GET MODEL INPUT TENSOR INDEX"
      continue
    fi

    # Get op output tensor names
    OP_OUT_NAMES=( $("${CIRCLEDUMP}" "${TESTCASE_FILE}.circle" | grep -P ' O T\(\d+:' | grep -oE '[^ ]+$') )
    if [[ $? -ne 0 ]]; then
      echo "FAILED TO GET OP OUTPUT TENSOR INDEX"
      continue
    fi
    declare -p OP_OUT_NAMES

    # Run minmax-embedder-data-gen
    RUNS=2
    for (( RUN=1; RUN<=RUNS; RUN++ )); do
      "${MINMAX_DATA_GEN}" --num_inputs ${#INPUT_NAMES[@]} --num_ops ${#OP_OUT_NAMES[@]} "${TESTCASE}.minmax"
      if [[ $? -ne 0 ]]; then
        echo "FAILED TO GENERATE MINMAX DATA"
        continue
      fi
    done

    # Run minmax-embedder
    "${MINMAX_EMBEDDER}" \
      --min_percentile 0 --max_percentile 100 \
      -o "${TESTCASE}.circle+minmax" \
      "${TESTCASE_FILE}.circle" \
      "${TESTCASE}.minmax"
    if [[ $? -ne 0 ]]; then
      echo "FAILED TO EMBED MINMAX INTO CIRCLE"
      continue
    fi

    # rm -f "${TESTCASE}.minmax"

    # Read min/max from circle+minmax
    MD_MIN=()
    MD_MAX=()
    for NAME in "${INPUT_NAMES[@]}"; do 
      MD_MIN+=( $("${CIRCLEDUMP}" "${TESTCASE}.circle+minmax" | grep -P "^T.*${NAME}$" -A 1 | tail -1 | grep -oP '(?<=min\()[+-]?[\d]+') )
      if [[ $? -ne 0 ]]; then
        echo "FAILED TO PARSE MODEL INPUT MIN FROM CIRCLE"
        continue
      fi
      MD_MAX+=( $("${CIRCLEDUMP}" "${TESTCASE}.circle+minmax" | grep -P "^T.*${NAME}$" -A 1 | tail -1 | grep -oP '(?<=max\()[+-]?[\d]+') )
      if [[ $? -ne 0 ]]; then
        echo "FAILED TO PARSE MODEL INPUT MAX FROM CIRCLE"
        continue
      fi
    done
    declare -p MD_MAX
    declare -p MD_MIN

    OP_MIN=()
    OP_MAX=()
    for NAME in "${OP_OUT_NAMES[@]}"; do 
      OP_MIN+=( $("${CIRCLEDUMP}" "${TESTCASE}.circle+minmax" | grep -P "^T.*${NAME}$" -A 1 | tail -1 | grep -oP '(?<=min\()[+-]?[\d]+') )
      if [[ $? -ne 0 ]]; then
        echo "FAILED TO PARSE OP MIN FROM CIRCLE"
        continue
      fi
      declare -p OP_MIN
      OP_MAX+=( $("${CIRCLEDUMP}" "${TESTCASE}.circle+minmax" | grep -P "^T.*${NAME}$" -A 1 | tail -1 | grep -oP '(?<=max\()[+-]?[\d]+') )
      if [[ $? -ne 0 ]]; then
        echo "FAILED TO PARSE OP MAX FROM CIRCLE"
        continue
      fi
      declare -p OP_MAX
    done

    # check model input 
    for i in "${!MD_MIN[@]}"; do
      # Be sure it is synced with minmax-embedder-data-gen
      EXPECTED_MIN=$((i*10))
      EXPECTED_MAX=$(((RUNS-1)*10000+i*10+7))
      if [[ "${MD_MIN[i]}" != "$EXPECTED_MIN" ]]; then
        echo "Min at model input $i does not equal."
        continue
      fi
      if [[ "${MD_MAX[i]}" != "$EXPECTED_MAX" ]]; then
        echo "Max at model input $i does not equal."
        continue
      fi
    done

    # check op output
    for i in "${!OP_MIN[@]}"; do
      # Be sure it is synced with minmax-embedder-data-gen
      EXPECTED_MIN=$((i*10))
      EXPECTED_MAX=$(((RUNS-1)*10000+i*10+7))
      if [[ "${OP_MIN[i]}" != "$EXPECTED_MIN" ]]; then
        echo "Min at op $i does not equal."
        continue
      fi
      if [[ "${OP_MAX[i]}" != "$EXPECTED_MAX" ]]; then
        echo "Max at op $i does not equal."
        continue
      fi
    done
    touch "${PASSED_TAG}"
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
