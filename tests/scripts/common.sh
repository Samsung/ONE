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

MY_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_PATH="$(cd ${MY_PATH}/../../ && pwd)"

# Install path on CI
INSTALL_PATH=$ROOT_PATH/Product/out
TEST_CACHE_PATH=$INSTALL_PATH/test/cache

function prepare_test_model()
{
  # Model download server setting
  if [[ -z "${MODELFILE_SERVER}" ]]; then
    echo "Model file server is not set. Try to use default setting."
  else
    echo "Model Server: ${MODELFILE_SERVER}"
  fi
  $INSTALL_PATH/test/onert-test prepare-model --cachedir=$TEST_CACHE_PATH
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
