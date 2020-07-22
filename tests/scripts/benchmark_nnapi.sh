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

BENCHMARK_DRIVER_BIN=
BENCHMARK_REPORT_DIR=
BENCHMARK_MODELS_FILE=
MODEL_TEST_ROOT_PATH=
TEST_OP="false"
BENCHMARK_MODEL_LIST="MODELS/inception_nonslim MODELS/inception_slim MODELS/mobilenet"
BACKEND_LIST="acl_cl acl_neon cpu" #TODO: accept this list as argument
EXECUTORS="Linear Parallel" #TODO: accept this list as argument

function Usage()
{
    echo "Usage: ./$0 --reportdir=. --driverbin=Product/out/bin/tflite_run"
}

for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --test_op)
            TEST_OP="true"
            ;;
        --driverbin=*)
            BENCHMARK_DRIVER_BIN=${i#*=}
            ;;
        --reportdir=*)
            BENCHMARK_REPORT_DIR=${i#*=}
            BENCHMARK_MODELS_FILE=$BENCHMARK_REPORT_DIR/benchmark_models.txt
            ;;
        --modelfilepath=*)
            TEST_LIST_PATH=${i#*=}
            MODEL_TEST_ROOT_PATH=$TEST_LIST_PATH/tests
            ;;
    esac
    shift
done

function get_benchmark_op_list()
{
    local TEST_DIRS="$@"
    local TESTS_TO_RUN=""

    if [[ $# -eq 0 ]]; then
        TEST_DIRS="."
    fi

    shift $#

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

function profile_for_he_shed()
{

    local REPORT_MODEL_DIR=$1
    local RUN_TEST_SH=$2
    local BENCHMARK_DRIVER_BIN=$3
    local MODEL=$4
    local PROFILING_RUN_CNT=$5

    export USE_SCHEDULER=1
    export PROFILING_MODE=1
    export EXECUTOR="Dataflow"
    export ONERT_LOG_ENABLE=1

    rm "exec_time.json" 2>/dev/null
    for ((j = 1 ; j <= $PROFILING_RUN_CNT ; j++)); do
        # Save the verbose log of each run
        LOG_FILE=$REPORT_MODEL_DIR/tflite_profiling_$j.txt

        print_with_dots "Profiling run #$j out of $PROFILING_RUN_CNT"

        $RUN_TEST_SH --driverbin=$BENCHMARK_DRIVER_BIN $MODEL > $LOG_FILE 2>&1
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "Profiling $MODEL aborted in run#$j... exit code: $RET"xX
            exit $RET
        fi
        echo "finished"
        # Save the exec_time.json of each run
        cp "exec_time.json" $REPORT_MODEL_DIR/"exec_time_$j.json"
    done
    unset USE_SCHEDULER PROFILING_MODE EXECUTOR ONERT_LOG_ENABLE
}

function run_with_he_scheduler()
{
    local REPORT_MODEL_DIR=$1
    local RUN_TEST_SH=$2
    local BENCHMARK_DRIVER_BIN=$3
    local MODEL=$4
    local EXECUTOR=$5

    LOG_FILE=$REPORT_MODEL_DIR/tflite_onert_with_he_scheduler_in_$EXECUTOR.txt
    export EXECUTOR=$EXECUTOR
    export GRAPH_DOT_DUMP=1
    export USE_SCHEDULER=1
    export ONERT_LOG_ENABLE=1

    print_with_dots "TFLite onert $EXECUTOR with HEScheduler"

    RESULT=$(get_result_of_benchmark_test $RUN_TEST_SH $BENCHMARK_DRIVER_BIN $MODEL $LOG_FILE)
    echo "$RESULT ms"

    mv "after_lower.dot" $REPORT_MODEL_DIR/"after_lower_$EXECUTOR.dot"
    unset EXECUTOR GRAPH_DOT_DUMP USE_SCHEDULER ONERT_LOG_ENABLE
}

function run_onert_with_all_config()
{
    local MODEL=$1
    local REPORT_MODEL_DIR=$2
    local PAUSE_TIME_IN_SEC=$3
    local BENCHMARK_DRIVER_BIN=$4
    local EXECUTORS=$5
    local BACKEND_LIST=$6

    export USE_NNAPI=1

    # Run profiler BACKEND_CNT+1 times: on each run of the first BACKEND_CNT runs it will
    #     collect metrics for one unmeasured backend. On the last run metrics for data transfer
    PROFILING_RUN_CNT=1
    BACKENDS_TO_USE=
    for backend in $BACKEND_LIST; do
        BACKENDS_TO_USE+=$backend';'
        ((++PROFILING_RUN_CNT))
    done
    export BACKENDS=$BACKENDS_TO_USE
    if [ "$TEST_OP" == "false" ]; then
        profile_for_he_shed $REPORT_MODEL_DIR $BENCHMARK_DRIVER_BIN $MODEL $PROFILING_RUN_CNT
    fi

    for executor in $EXECUTORS; do
        export EXECUTOR=$executor
        if [ "$TEST_OP" == "false" ]; then
            run_with_he_scheduler $REPORT_MODEL_DIR $BENCHMARK_DRIVER_BIN $MODEL $executor
        fi
        for backend in $BACKEND_LIST; do
            export OP_BACKEND_ALLOPS=$backend
            run_benchmark_and_print "tflite_onert_"$executor"_executor_$backend" "TFLite onert $executor Executor $backend"\
                                    $MODEL $REPORT_MODEL_DIR 0 $BENCHMARK_DRIVER_BIN
        done
    done
    unset USE_NNAPI EXECUTOR OP_BACKEND_ALLOPS BACKENDS
}

function run_benchmark_test()
{
    local LOG_FILE=
    local RESULT_FILE=
    local RESULT=
    local REPORT_MODEL_DIR=

    export COUNT=5
    export ONERT_LOG_ENABLE=1
    echo
    echo "============================================"
    echo
    date +'%Y-%m-%d %H:%M:%S %s'
    echo
    local i=0
    for MODEL in $BENCHMARK_MODEL_LIST; do

        STATUS="enabled"
        if [ "$TEST_OP" == "true" ]; then
            source $MODEL_TEST_ROOT_PATH/$MODEL/config.sh
        fi

        # Skip 'disabled' tests
        if [ $(tr '[:upper:]' '[:lower:]' <<< "$STATUS") == "disabled" ]; then
            continue
        fi

        echo "Benchmark test with `basename $BENCHMARK_DRIVER_BIN` & `echo $MODEL`"
        echo $MODEL >> $BENCHMARK_MODELS_FILE

        REPORT_MODEL_DIR=$BENCHMARK_REPORT_DIR/$MODEL
        mkdir -p $REPORT_MODEL_DIR

        # TFLite+CPU
        unset USE_NNAPI
        run_benchmark_and_print "tflite_cpu" "TFLite CPU" $MODEL $REPORT_MODEL_DIR 0 $BENCHMARK_DRIVER_BIN

        # run onert
        if [ "$TEST_OP" == "true" ]; then
          # Operation test don't need to test each scheduler
          run_onert_with_all_config $MODEL $REPORT_MODEL_DIR 0 $BENCHMARK_DRIVER_BIN "Linear" "$BACKEND_LIST"
        else
          run_onert_with_all_config $MODEL $REPORT_MODEL_DIR 0 $BENCHMARK_DRIVER_BIN "$EXECUTORS" "$BACKEND_LIST"
        fi

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

if [ "$TEST_OP" == "true" ]; then
    get_benchmark_op_list
fi

rm -rf $BENCHMARK_MODELS_FILE

echo ""
# print the result AND append to log file
run_benchmark_test 2>&1 | tee -a onert_benchmarks.txt
echo ""
