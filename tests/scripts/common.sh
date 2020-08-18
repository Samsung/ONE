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

function get_result_of_benchmark_test()
{
    local DRIVER_BIN=$1
    local MODEL=$2
    local LOG_FILE=$3

    local RET=0
    $MY_PATH/models/run_test.sh --driverbin="$DRIVER_BIN  -r 5 -w 3" $MODEL > $LOG_FILE 2>&1
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "Testing $MODEL aborted... exit code: $RET"
        exit $RET
    fi

    local RESULT=`grep -E '^- MEAN ' $LOG_FILE | awk '{print $4}'`
    echo "$RESULT"
}

function print_result_of_benchmark_test()
{
    local NAME=$1
    local RESULT=$2
    local RESULT_FILE=$3

    echo "$NAME $RESULT" > $RESULT_FILE
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


function run_benchmark_and_print()
{
    local WRITE_FILE_NAME=$1
    local MSG=$2
    local MODEL=$3
    local REPORT_MODEL_DIR=$4
    local PAUSE_TIME_IN_SEC=$5
    local DRIVER_BIN=$6
    local BENCHMARK_RUN_TEST_SH=$7

    LOG_FILE=$REPORT_MODEL_DIR/$WRITE_FILE_NAME.txt
    RESULT_FILE=$REPORT_MODEL_DIR/$WRITE_FILE_NAME.result
    print_with_dots $MSG
    RESULT=$(get_result_of_benchmark_test $DRIVER_BIN $MODEL $LOG_FILE)
    echo "$RESULT ms"
    print_result_of_benchmark_test "$MSG" "$RESULT" $RESULT_FILE
    sleep $PAUSE_TIME_IN_SEC
}
