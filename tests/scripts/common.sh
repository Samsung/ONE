#!/bin/bash
#
# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_PATH="$(cd ${CURRENT_PATH}/../../ && pwd)"

# Install path on CI
INSTALL_PATH=${INSTALL_PATH:-$ROOT_PATH}
PATH=$INSTALL_PATH/bin:$PATH
TEST_CACHE_PATH=$INSTALL_PATH/test/cache

function PrepareTestModel()
{
  # Model download server setting
  if [[ -z "${MODELFILE_SERVER}" ]]; then
    echo "Model file server is not set. Try to use default setting."
  else
    echo "Model Server: ${MODELFILE_SERVER}"
  fi
  onert-test prepare-model --cachedir=$TEST_CACHE_PATH
}

function get_result_of_benchmark_test()
{
    local DRIVER_BIN=$1
    local MODEL=$2
    local LOG_FILE=$3

    local RET=0
    $INSTALL_PATH/test/models/run_test.sh --driverbin="$DRIVER_BIN -r 5 -w 3" --cachedir=$TEST_CACHE_PATH $MODEL > $LOG_FILE 2>&1
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "Testing $MODEL aborted... exit code: $RET"
        exit $RET
    fi

    local RESULT=`grep -E '^- MEAN ' $LOG_FILE | awk '{print $4}'`
    echo "$RESULT"
}

function print_with_dots()
{
    PRINT_WIDTH=45
    local MSG="$@"
    pad=$(printf '%0.1s' "."{1..45})
    padlength=$((PRINT_WIDTH- ${#MSG}))
    printf '%s' "$MSG"
    printf '%*.*s ' 0 $padlength "$pad"
}

# $1: (required) backend
# $2: (required) framework list file relative path from nnfw root directory
#                pass empty string if there is no skiplist
# $3: (required) relative path to report from nnfw root directory
function TFLiteModelVerification()
{
  [[ $# -ne 3 ]] && echo "Invalid function argument setting" && exit 1

  export BACKENDS=$1
  if [[ "$2" == "" ]]; then
    onert-test verify-tflite \
      --reportdir=$3
  else
    onert-test verify-tflite \
      --list=$2 \
      --reportdir=$3
  fi
  unset BACKENDS
}

# $1: (required) backend
# $2: (required) nnapi gtest skiplist file relative path from nnfw root directory
#                pass empty string if there is no test list
# $3: (required) relative path for report from nnfw root directory
function NNAPIGTest()
{
  [[ $# -ne 3 ]] && echo "Invalid function argument setting" && exit 1

  # Backup original nnapi_gtest.skip
  # TODO Pass skiplist to test-driver.sh
  SKIPLIST_FILE="${INSTALL_PATH}/nnapi-gtest/nnapi_gtest.skip"
  BACKUP_FILE="${SKIPLIST_FILE}.backup"
  if [[ "$2" != "" ]]; then
    cp ${SKIPLIST_FILE} ${BACKUP_FILE}
    cp $2 ${SKIPLIST_FILE}
  fi

  export BACKENDS=$1
  onert-test unittest \
    --reportdir=$3 \
    --unittestdir=$INSTALL_PATH/nnapi-gtest
  unset BACKENDS

  # TODO Pass skiplist to test-driver.sh
  # Restore original nnapi_gtest.skip
  if [[ "$2" != "" ]]; then
    cp ${BACKUP_FILE} ${SKIPLIST_FILE}
    rm ${BACKUP_FILE}
  fi
}

# $1: (require) backend
# $2: (require) list
function NNPackageTest()
{
  [[ $# -ne 2 ]] && echo "Invalid function argument setting" && exit 1

  echo "[Package Test] Run $1 backend nnpackage test"

  EXITCODE=0
  PKG_LIST=$(cat $2)
  for f in ${PKG_LIST}
  do
    for entry in "nnpkg-tcs"/$f; do
      if [ -e $entry ]; then
        BACKENDS="$1" onert-test nnpkg-test -d -i nnpkg-tcs $(basename "$entry")
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
}
