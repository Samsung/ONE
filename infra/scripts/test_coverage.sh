#!/bin/bash

# Test suite: ${ARCHIVE_PATH}/coverage-suite.tar.gz
# NNPackage test suite: ${ARCHIVE_PATH}/nnpkg-test-suite.tar.gz (optional)

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

pushd $ROOT_PATH > /dev/null

NNAS_WORKSPACE=${NNAS_WORKSPACE:-build}
if [[ -z "${ARCHIVE_PATH}" ]]; then
  ARCHIVE_PATH=${NNAS_WORKSPACE}/archive
  echo "Default archive directory including nncc package and resources: ${ARCHIVE_PATH}"
fi

tar -zxf ${ARCHIVE_PATH}/coverage-suite.tar.gz -C ./

CheckTestPrepared

if [[ ! -e $ROOT_PATH/tests/scripts/build_path_depth.txt ]]; then
  echo "Cannot find prefix strip file"
  exit 1
fi
export GCOV_PREFIX_STRIP=`cat $ROOT_PATH/tests/scripts/build_path_depth.txt`

TENSOR_LOGGING=trace_log.txt ./infra/scripts/test_ubuntu_runtime.sh --backend acl_cl --nnapi-frontend
./infra/scripts/test_ubuntu_runtime.sh --backend acl_neon
./infra/scripts/test_ubuntu_runtime.sh --backend cpu

# Enable all logs (mixed backend)
ONERT_LOG_ENABLE=1 GRAPH_DOT_DUMP=1 ./infra/scripts/test_ubuntu_runtime_mixed.sh
# Enable trace event (acl_cl default backend)
export TRACING_MODE=1
TFLiteModelVerification "acl_cl" "Product/out/test/list/tflite_comparator.armv7l.acl_cl.list" "report/acl_cl/trace"
unset TRACING_MODE

# nnpackage test suite
if [[ -e ${ARCHIVE_PATH}/nnpkg-test-suite.tar.gz ]]; then
  tar -zxf ${ARCHIVE_PATH}/nnpkg-test-suite.tar.gz -C ./
  ./infra/scripts/test_arm_nnpkg.sh
fi

# Pack coverage test data: coverage-data.tar.gz
find Product -type f \( -iname *.gcda -or -iname *.gcno \) > include_lists.txt
tar -zcf ${ARCHIVE_PATH}/coverage-data.tar.gz -T include_lists.txt
rm -rf include_lists.txt

popd > /dev/null
