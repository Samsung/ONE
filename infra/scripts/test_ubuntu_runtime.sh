#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKEND="cpu"
TEST_ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')
TEST_OS="linux"
TFLITE_LOADER="0"
LINEAR_ONLY="0"

function Usage()
{
  echo "Usage: $0 $(basename ${BASH_SOURCE[0]}) [OPTIONS]"
  echo ""
  echo "Options:"
  echo "      --backend <BACKEND>     Runtime backend to test (default: ${BACKEND})"
  echo "      --tflite-loader         Enable TFLite Loader test"
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
    --tflite-loader)
      TFLITE_LOADER="1"
      shift
      ;;
    --linear-only)
      LINEAR_ONLY="1"
      shift
      ;;
    *)
      # Ignore
      shift
      ;;
  esac
done

CheckTestPrepared
echo "[[ ${TEST_ARCH}-${TEST_OS}: ${BACKEND} backend test ]]"
UNITTEST_SKIPLIST="Product/out/unittest/nnapi_gtest.skip.${TEST_ARCH}-${TEST_OS}.${BACKEND}"
FRAMEWORK_TESTLIST="tests/scripts/list/frameworktest_list.${TEST_ARCH}.${BACKEND}.txt"
REPORT_BASE="report/${BACKEND}"
EXECUTORS=("Linear" "Dataflow" "Parallel")
if [ $LINEAR_ONLY = "1" ]; then
  EXECUTORS=("Linear")
fi

for EXECUTOR in "${EXECUTORS[@]}";
do
  echo "[EXECUTOR]: ${EXECUTOR}"
  export EXECUTOR="${EXECUTOR}"
  Unittests "${BACKEND}" "${UNITTEST_SKIPLIST}" "${REPORT_BASE}/${EXECUTOR}"
  TFLiteModelVerification "${BACKEND}" "${FRAMEWORK_TESTLIST}" "${REPORT_BASE}/${EXECUTOR}"
  unset EXECUTOR
done

# Current support acl_cl backend testlist only
# TODO Support more backends
TFLITE_LOADER_TESTLIST="tests/scripts/list/tflite_loader_list.${TEST_ARCH}.txt"
if [[ $TFLITE_LOADER = "1" ]]; then
  TFLiteLoaderTest "${BACKEND}" "${TFLITE_LOADER_TESTLIST}" "${REPORT_BASE}/loader/${EXECUTOR}"

  # Test custom op
  pushd ${ROOT_PATH} > /dev/null
  ./Product/out/tests/FillFrom_runner
  popd > /dev/null
fi
