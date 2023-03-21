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

PrepareTestModel

if [[ ! -e $INSTALL_PATH/test/scripts/build_path_depth.txt ]]; then
  echo "Cannot find prefix strip file"
  exit 1
fi
export GCOV_PREFIX_STRIP=`cat $INSTALL_PATH/test/scripts/build_path_depth.txt`

TENSOR_LOGGING=trace_log.txt $INSTALL_PATH/test/scripts/test_ubuntu_runtime.sh --backend acl_cl
$INSTALL_PATH/test/scripts/test_ubuntu_runtime.sh --backend acl_neon
$INSTALL_PATH/test/scripts/test_ubuntu_runtime.sh --backend cpu

# Enable all logs (mixed backend)
ONERT_LOG_ENABLE=1 GRAPH_DOT_DUMP=1 $INSTALL_PATH/test/scripts/test_ubuntu_runtime_mixed.sh
# Enable trace event (acl_cl default backend)
export TRACE_FILEPATH=trace.json
TFLiteModelVerification "acl_cl" "$INSTALL_PATH/test/list/tflite_comparator.armv7l.acl_cl.list" "report/acl_cl/trace"
unset TRACE_FILEPATH

# nnpackage test suite
if [[ -e ${ARCHIVE_PATH}/nnpkg-test-suite.tar.gz ]]; then
  tar -zxf ${ARCHIVE_PATH}/nnpkg-test-suite.tar.gz -C ./
  $INSTALL_PATH/test/scripts/test_arm_nnpkg.sh
fi

# Pack coverage test data: coverage-data.tar.gz
rm -rf $INSTALL_PATH/gcov-data
mkdir -p $INSTALL_PATH/gcov-data
find $INSTALL_PATH -type f \( -iname *.gcda -or -iname *.gcno \) -exec cp {} $INSTALL_PATH/gcov-data/. \;
tar -zcf ${ARCHIVE_PATH}/coverage-data.tar.gz -C $INSTALL_PATH/gcov-data .

popd > /dev/null
