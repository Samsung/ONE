#!/bin/bash

# Don't run this script
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && echo "Please don't execute ${BASH_SOURCE[0]}, source it" && return

# Global variable
# CURRENT_PATH: infra/scripts directory absolute path
# ROOT_PATH: nnfw root directory absolute path

# Functions
#
# CheckTestPrepared
#   Check environment variable setting to run test
#
# TFLiteModelVerification $1 $2 $3
#   Run ./tests/scripts/test-driver.sh script verification test
#
# Unittests $1 $2 $3
#   Run ./tests/scripts/test-driver.sh script unittest
#
# NNPackageTest $1 $2
#   Run [INSTALL_PATH]/test/onert-test nnpkg-test command

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd ${CURRENT_PATH}/../../ && pwd)"

# Install path on CI
INSTALL_PATH=$ROOT_PATH/Product/out

function CheckTestPrepared()
{
  # Model download server setting
  if [[ -z "${MODELFILE_SERVER}" ]]; then
    echo "[WARNING] Model file server is not set"
    echo "          Try to use pre-downloaed model"
  else
    echo "Model Server: ${MODELFILE_SERVER}"
  fi
}

# $1: (required) backend
# $2: (required) framework list file relative path from nnfw root directory
#                pass empty string if there is no skiplist
# $3: (required) relative path to report from nnfw root directory
function TFLiteModelVerification()
{
  [[ $# -ne 3 ]] && echo "Invalid function argument setting" && exit 1

  pushd ${ROOT_PATH} > /dev/null

  export BACKENDS=$1
  if [[ "$2" == "" ]]; then
    ./tests/scripts/test-driver.sh \
      --reportdir=$ROOT_PATH/$3 \
      --verification \
      .
  else
    ./tests/scripts/test-driver.sh \
      --frameworktest_list_file=$2 \
      --reportdir=$ROOT_PATH/$3 \
      --verification \
      .
  fi
  unset BACKENDS

  popd > /dev/null
}

# $1: (required) backend
# $2: (required) unittest skiplist file relative path from nnfw root directory
#                pass empty string if there is no test list
# $3: (required) relative path for report from nnfw root directory
function Unittests()
{
  [[ $# -ne 3 ]] && echo "Invalid function argument setting" && exit 1

  pushd ${ROOT_PATH} > /dev/null

  # Backup original nnapi_gtest.skip
  # TODO Pass skiplist to test-driver.sh
  SKIPLIST_FILE="${ROOT_PATH}/Product/out/unittest/nnapi_gtest.skip"
  BACKUP_FILE="${SKIPLIST_FILE}.backup"
  if [[ "$2" != "" ]]; then
    cp ${SKIPLIST_FILE} ${BACKUP_FILE}
    cp ${ROOT_PATH}/$2 ${SKIPLIST_FILE}
  fi

  export BACKENDS=$1
  ./tests/scripts/test-driver.sh \
    --reportdir=$ROOT_PATH/$3 \
    --unittest \
    .
  unset BACKENDS

  # TODO Pass skiplist to test-driver.sh
  # Restore original nnapi_gtest.skip
  if [[ "$2" != "" ]]; then
    cp ${BACKUP_FILE} ${SKIPLIST_FILE}
    rm ${BACKUP_FILE}
  fi

  popd > /dev/null
}

# $1: (require) backend
# $2: (require) list
function NNPackageTest()
{
  [[ $# -ne 2 ]] && echo "Invalid function argument setting" && exit 1

  pushd ${ROOT_PATH} > /dev/null

  echo "[Package Test] Run $1 backend nnpackage test"

  EXITCODE=0
  PKG_LIST=$(cat $2)
  for f in ${PKG_LIST}
  do
    for entry in "nnpkg-tcs"/$f; do
      if [ -e $entry ]; then
        BACKENDS="$1" $INSTALL_PATH/test/onert-test nnpkg-test -d -i nnpkg-tcs $(basename "$entry")
      fi
    done
    EXITCODE_F=$?

    if [ ${EXITCODE_F} -ne 0 ]; then
      EXITCODE=${EXITCODE_F}
    fi
  done

  if [ ${EXITCODE} -ne 0 ]; then
    exit ${EXITCODE}
  fi

  popd > /dev/null
}

# $1: (required) backend
# $2: (required) test list file relative path from nnfw root directory
#                pass empty string if there is no skiplist
# $3: (required) relative path to report from nnfw root directory
function TFLiteLoaderTest()
{
  [[ $# -ne 3 ]] && echo "TFLiteLoaderTest: Invalid function argument setting" && exit 1

  pushd ${ROOT_PATH} > /dev/null

  export BACKENDS=$1
  if [[ "$2" == "" ]]; then
    ./tests/scripts/test-driver.sh \
      --frameworktest \
      --framework_driverbin="$ROOT_PATH/Product/out/bin/tflite_loader_test_tool" \
      --reportdir=$ROOT_PATH/$3
      .
  else
    ./tests/scripts/test-driver.sh \
      --frameworktest \
      --framework_driverbin="$ROOT_PATH/Product/out/bin/tflite_loader_test_tool" \
      --frameworktest_list_file=tests/scripts/list/tflite_loader_list.${TEST_ARCH}.txt \
      --reportdir=$ROOT_PATH/$3
  fi
  unset BACKENDS

  popd > /dev/null
}
