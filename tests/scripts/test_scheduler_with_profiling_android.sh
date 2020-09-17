#!/system/bin/sh
#
# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#
# How to run benchmark testing
#
# This script is copy of test_scheduler_with_profiling.sh for Android.
# As Android does not provide bash, this with models/run_test_android.sh
# and common_android.sh, three scripts are modified for Android benchmark
# testing using Android shell.
# Test models are downloaded into models folder but as Android also doesn't
# provide downloading in shell script, user should push downloaded models
# to Android device also.
#
# 1. To download test models,
#    run test_scheduler_with_profiling.sh from in Ubuntu/ARM device
# 2. You will have download models in tests/scripts/models/cache folder
# 3. Build for OneRT for Android
# 4. Copy files
#    adb shell mkdir -p /data/local/tmp/Product/report/benchmark
#    adb push tests /data/local/tmp/.
#    adb push Product/aarch64-android.release/out /data/local/tmp/Product/.
#
# 5. Run benchmark inside Android shell
#    export LD_LIBRARY_PATH=/data/local/tmp/Product/out/lib
#    cd /data/local/tmp
#    sh /data/local/tmp/tests/scripts/test_scheduler_with_profiling_android.sh
#

MY_PATH="$( cd "$( dirname "$0" )" && pwd )"

SHELL_CMD=/system/bin/sh

source $MY_PATH/common_android.sh

BACKEND_CNT=3
# Run profiler BACKEND_CNT+1 times: on each run of the first BACKEND_CNT runs it will
#     collect metrics for one unmeasured backend. On the last run metrics for data transfer
PROFILING_RUN_CNT=$((BACKEND_CNT+1))
TEST_DRIVER_DIR="$( cd "$( dirname "$0" )" && pwd )"

ARTIFACT_PATH="$TEST_DRIVER_DIR/../.."
BENCHMARK_DRIVER_BIN=$ARTIFACT_PATH/Product/out/bin/tflite_run
REPORT_DIR=$ARTIFACT_PATH/report
RUN_TEST_SH=$ARTIFACT_PATH/tests/scripts/models/run_test_android.sh
BENCHMARK_MODEL_LIST="MODELS/inception_nonslim MODELS/inception_slim MODELS/mobilenet"

if [ ! -e "$RUN_TEST_SH" ]; then
    echo "Cannot find $RUN_TEST_SH"
    exit 1
fi

BENCHMARK_REPORT_DIR=$REPORT_DIR/benchmark
BENCHMARK_MODELS_FILE=$BENCHMARK_REPORT_DIR/benchmark_models.txt

function run_without_sched()
{
    local RESULT_SCH_INT=$1
    local REPORT_MODEL_DIR=$2
    local MODEL=$3
    local EXECUTOR=$4
    local BACKEND=$5

    #LOG_FILE=$REPORT_MODEL_DIR/tflite_${EXECUTOR,,}_$BACKEND.txt
    LOG_FILE=$REPORT_MODEL_DIR/tflite_$EXECUTOR_$BACKEND.txt
    export OP_BACKEND_ALLOPS=$BACKEND
    export EXECUTOR=$EXECUTOR

    print_with_dots "$EXECUTOR $BACKEND without scheduler"

    RESULT=$(get_result_of_benchmark_test $BENCHMARK_DRIVER_BIN $MODEL $LOG_FILE)

    # printf -v RESULT_INT '%d' $RESULT 2>/dev/null
    RESULT_I=$(printf "%.0f" $RESULT)
    RESULT_INT=$(expr $RESULT_I)
    PERCENTAGE=$((100 - RESULT_SCH_INT * 100 / RESULT_INT))
    echo "$RESULT ms. Parallel scheduler is $PERCENTAGE % faster"
}

function run_benchmark_test()
{
    local LOG_FILE=
    local RESULT=
    local REPORT_MODEL_DIR=

    export COUNT=5
    echo "============================================"
    local i=0
    export USE_NNAPI=1
    export BACKENDS="acl_cl;acl_neon;cpu"
    # Remove metrics so that profiler can get metrics for operations
    #      with input&output sizes the same as the model
    rm "exec_time.json" 2>/dev/null
    for MODEL in $BENCHMARK_MODEL_LIST; do

        echo "Benchmark test with `basename $BENCHMARK_DRIVER_BIN` & `echo $MODEL`"
        echo $MODEL >> $BENCHMARK_MODELS_FILE

        REPORT_MODEL_DIR=$BENCHMARK_REPORT_DIR/scheduler_benchmark/$MODEL
        mkdir -p $REPORT_MODEL_DIR

##################################################################################
        # Get metrics by running profiler
##################################################################################
        export USE_SCHEDULER=1
        export PROFILING_MODE=1
        export EXECUTOR="Dataflow"
        export ONERT_LOG_ENABLE=1
        for j in 1 2 3 4; do # 1 to $PROFILING_RUN_CNT
            # Save the verbose log of each run
            LOG_FILE=$REPORT_MODEL_DIR/tflite_profiling_$j.txt

            print_with_dots "Profiling run #$j out of $PROFILING_RUN_CNT"

            $SHELL_CMD $RUN_TEST_SH --driverbin=$BENCHMARK_DRIVER_BIN $MODEL > $LOG_FILE 2>&1
            RET=$?
            if [[ $RET -ne 0 ]]; then
                echo "Profiling $MODEL aborted in run#$j... exit code: $RET"xX
                exit $RET
            fi
            echo "finished"
            # Save the exec_time.json of each run
            cp "exec_time.json" $REPORT_MODEL_DIR/"exec_time_$j.json"
        done
        unset ONERT_LOG_ENABLE


##################################################################################
        # Turn off profiling
##################################################################################
        export PROFILING_MODE=0

##################################################################################
        # Run ParallelExecutor with scheduler
##################################################################################
        LOG_FILE=$REPORT_MODEL_DIR/tflite_parallel_with_scheduler.txt
        export EXECUTOR="Parallel"
        export GRAPH_DOT_DUMP=1
        print_with_dots "Parallel with scheduler"

        RESULT=$(get_result_of_benchmark_test $BENCHMARK_DRIVER_BIN $MODEL $LOG_FILE)
        echo "$RESULT ms"

        # printf -v RESULT_SCH_INT '%d' $RESULT 2>/dev/null
        RESULT_I=$(printf "%.0f" $RESULT)
        RESULT_SCH_INT=$(expr $RESULT_I)

        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_parallel.dot"

##################################################################################
        # Run Linear executor with scheduler
##################################################################################
        LOG_FILE=$REPORT_MODEL_DIR/tflite_linear_with_scheduler.txt
        export EXECUTOR="Linear"
        export GRAPH_DOT_DUMP=1
        print_with_dots "Linear with scheduler"

        RESULT=$(get_result_of_benchmark_test $BENCHMARK_DRIVER_BIN $MODEL $LOG_FILE)

        # printf -v RESULT_INT '%d' $RESULT 2>/dev/null
        RESULT_I=$(printf "%.0f" $RESULT)
        RESULT_INT=$(expr $RESULT_I)

        PERCENTAGE=$((100 - $RESULT_SCH_INT * 100 / $RESULT_INT))

        echo "$RESULT ms. Parallel scheduler is $PERCENTAGE % faster"

        # Remove metrics so that for next model in profiler can get metrics
        #   for operations with input&output sizes the same as the model
        mv "exec_time.json" $REPORT_MODEL_DIR
        # Save the dot graph
        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_linear.dot"
        unset GRAPH_DOT_DUMP

##################################################################################
        # Turn off scheduler
##################################################################################
        export USE_SCHEDULER=0

        # Run LinearExecutor on acl_cl without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Linear" "acl_cl"

        # Run LinearExecutor on acl_neon without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Linear" "acl_neon"

        # Run LinearExecutor on cpu without scheduler
        # run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Linear" "cpu"

        # Run ParallelExecutor on acl_cl without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Parallel" "acl_cl"

        # Run ParallelExecutor on acl_neon without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Parallel" "acl_neon"

        # Run ParallelExecutor on cpi without scheduler
        # run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Parallel" "cpu"

        if [[ $i -ne $(echo $BENCHMARK_MODEL_LIST | wc -w)-1 ]]; then
            echo ""
        fi
        i=$((i+1))

        unset USE_SCHEDULER
        unset PROFILING_MODE
        unset EXECUTOR
        unset OP_BACKEND_ALLOPS
    done
    unset BACKENDS
    echo "============================================"
    unset COUNT
    unset USE_NNAPI

}

echo ""
run_benchmark_test
echo ""
