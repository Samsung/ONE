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

source $MY_PATH/common.sh

# Caution: DO NOT USE "pipefail"
#          We should run test all operators

ONERT_DRIVER_BIN=$INSTALL_PATH/bin/onert_run
TFLITE_DRIVER_BIN=$INSTALL_PATH/bin/tflite_run
REPORT_DIR=$ROOT_PATH/report
BENCHMARK_REPORT_DIR=$REPORT_DIR/benchmark_op
BENCHMARK_MODELS_FILE=$BENCHMARK_REPORT_DIR/benchmark_models.txt
MODEL_TEST_ROOT_PATH=$INSTALL_PATH/test/models/tflite
BENCHMARK_MODEL_LIST=
BACKEND_LIST="acl_cl acl_neon cpu"
TEST_DIRS="."

function Usage()
{
    echo "Usage: ${BASH_SOURCE[0]} [OPTIONS]"
    echo ""
    echo "Options:"
    echo "      --backends=STRING       Backends to test. (default='$BACKEND_LIST')"
    echo "      --list=FILE             List file to test. Test all if list option is not passed"
}

for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --list=*)
            TEST_LIST_PATH=${i#*=}
            TEST_DIRS=$(grep -v '#' $TEST_LIST_PATH | tr '\n' ' ' )
            ;;
        --backends=*)
            BACKEND_LIST=${i#*=}
            ;;
    esac
    shift
done

function get_benchmark_op_list()
{
    local TESTS_TO_RUN=""

    pushd $MODEL_TEST_ROOT_PATH > /dev/null
    for DIR in $TEST_DIRS; do
        if [ -d "$DIR" ]; then
            TESTS_FOUND=$(find "$DIR" -type f -name 'config.sh' -exec dirname {} \;| sed 's|^./||' | grep -v '^MODELS/' | sort)
            TESTS_TO_RUN="$TESTS_TO_RUN $TESTS_FOUND"
        fi
    done
    popd > /dev/null

    BENCHMARK_MODEL_LIST=$(echo "${TESTS_TO_RUN}")
}

function run_benchmark_and_print()
{

    local WRITE_FILE_NAME=$1
    local MSG=$2
    local MODEL=$3
    local REPORT_MODEL_DIR=$4
    local DRIVER_BIN=$5

    LOG_FILE=$REPORT_MODEL_DIR/$WRITE_FILE_NAME.txt
    RESULT_FILE=$REPORT_MODEL_DIR/$WRITE_FILE_NAME.result
    print_with_dots $MSG
    RESULT=$(get_result_of_benchmark_test $DRIVER_BIN $MODEL $LOG_FILE)
    echo "$RESULT ms"
    echo "$MSG $RESULT" > $RESULT_FILE
}

function run_onert_with_all_config()
{
    local MODEL=$1
    local REPORT_MODEL_DIR=$2
    local BENCHMARK_DRIVER_BIN=$3

    # Run profiler BACKEND_CNT+1 times: on each run of the first BACKEND_CNT runs it will
    #     collect metrics for one unmeasured backend. On the last run metrics for data transfer
    PROFILING_RUN_CNT=1
    BACKENDS_TO_USE=
    for backend in $BACKEND_LIST; do
        BACKENDS_TO_USE+=$backend';'
        ((++PROFILING_RUN_CNT))
    done
    export EXECUTOR="Linear"
    for backend in $BACKEND_LIST; do
        export BACKENDS=$backend
        run_benchmark_and_print "onert_$backend" "ONERT-${backend^^}"\
                                $MODEL $REPORT_MODEL_DIR $BENCHMARK_DRIVER_BIN
    done
    unset EXECUTOR BACKENDS
}

function run_benchmark_test()
{
    local LOG_FILE=
    local RESULT_FILE=
    local RESULT=
    local REPORT_MODEL_DIR=

    export COUNT=5
    echo
    echo "============================================"
    echo
    date +'%Y-%m-%d %H:%M:%S %s'
    echo
    local i=0
    for MODEL in $BENCHMARK_MODEL_LIST; do

        STATUS="enabled"
        source $MODEL_TEST_ROOT_PATH/$MODEL/config.sh

        # Skip 'disabled' tests
        if [ $(tr '[:upper:]' '[:lower:]' <<< "$STATUS") == "disabled" ]; then
            continue
        fi

        echo "Benchmark test `echo $MODEL`"
        echo $MODEL >> $BENCHMARK_MODELS_FILE

        REPORT_MODEL_DIR=$BENCHMARK_REPORT_DIR/$MODEL
        mkdir -p $REPORT_MODEL_DIR

        # TFLite+CPU
        run_benchmark_and_print "tflite_cpu" "TFLite-CPU" $MODEL $REPORT_MODEL_DIR $TFLITE_DRIVER_BIN

        # run onert
        # Operation test don't need to test each scheduler
        run_onert_with_all_config $MODEL $REPORT_MODEL_DIR $ONERT_DRIVER_BIN

        if [[ $i -ne $(echo $BENCHMARK_MODEL_LIST | wc -w)-1 ]]; then
            echo ""
        fi
        i=$((i+1))
    done
    echo "============================================"
    unset COUNT
}

if [ ! -e "$BENCHMARK_REPORT_DIR" ]; then
    mkdir -p $BENCHMARK_REPORT_DIR
fi

get_benchmark_op_list

rm -rf $BENCHMARK_MODELS_FILE

# Model download server setting
prepare_test_model

echo ""
# print the result AND append to log file
run_benchmark_test 2>&1 | tee -a $REPORT_DIR/onert_benchmarks.txt
echo ""

# Make json file.
# functions to fill json with benchmark results
source $MY_PATH/print_to_json.sh
print_to_json $BENCHMARK_REPORT_DIR $REPORT_DIR "benchmark_op_result.json"
