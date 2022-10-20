#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: ${TEST_ARCH:=$(uname -m | tr '[:upper:]' '[:lower:]')}
BACKEND="cpu"
TEST_OS="linux"
TEST_PLATFORM="$TEST_ARCH-$TEST_OS"
LINEAR_ONLY="0"
RUN_INTERP="0"

function Usage()
{
  echo "Usage: $0 $(basename ${BASH_SOURCE[0]}) [OPTIONS]"
  echo ""
  echo "Options:"
  echo "      --backend <BACKEND>     Runtime backend to test (default: ${BACKEND})"
  echo "      --linear-only           Use Linear executor only"
}

while [[ $# -gt 0 ]]
do
  arg="$1"
  case $arg in
    -h|--help|help)
      Usage
      exit 0
      ;;
    --backend)
      BACKEND=$(echo $2 | tr '[:upper:]' '[:lower:]')
      shift 2
      ;;
    --backend=*)
      BACKEND=$(echo ${1#*=} | tr '[:upper:]' '[:lower:]')
      shift
      ;;
    --linear-only)
      LINEAR_ONLY="1"
      shift
      ;;
    --interp)
      RUN_INTERP="1"
      shift;
      ;;
    *)
      # Ignore
      shift
      ;;
  esac
done

CheckTestPrepared

if [ $RUN_INTERP = "1" ]; then
  TEST_PLATFORM="noarch"
  TEST_ARCH="noarch"
  BACKEND="interp"
  echo "[[ Interpreter test ]]"
else
  echo "[[ ${TEST_PLATFORM}: ${BACKEND} backend test ]]"
fi

UNITTEST_SKIPLIST="Product/out/nnapi-gtest/nnapi_gtest.skip.${TEST_PLATFORM}.${BACKEND}"
TFLITE_TESTLIST="Product/out/test/list/tflite_comparator.${TEST_ARCH}.${BACKEND}.list"
REPORT_BASE="report/${BACKEND}"
EXECUTORS=("Linear" "Dataflow" "Parallel")

if [ $LINEAR_ONLY = "1" ]; then
  EXECUTORS=("Linear")
fi
if [ $RUN_INTERP = "1" ]; then
  EXECUTORS=("Interpreter")
fi

for EXECUTOR in "${EXECUTORS[@]}";
do
  echo "[EXECUTOR]: ${EXECUTOR}"
  REPORT_PATH="${REPORT_BASE}/${EXECUTOR}"

  if [ $EXECUTOR = "Interpreter" ]; then
    export DISABLE_COMPILE=1
    BACKEND=""
  else
    export EXECUTOR="${EXECUTOR}"
  fi

  NNAPIGTest "${BACKEND}" "${UNITTEST_SKIPLIST}" "${REPORT_PATH}"
  TFLiteModelVerification "${BACKEND}" "${TFLITE_TESTLIST}" "${REPORT_PATH}"

  if [ $EXECUTOR = "Interpreter" ]; then
    unset DISABLE_COMPILE
  else
    unset EXECUTOR
  fi
done
