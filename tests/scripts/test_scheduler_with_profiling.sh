#!/bin/bash

MY_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $MY_PATH/common.sh

BACKEND_CNT=3
# Run profiler BACKEND_CNT+1 times: on each run of the first BACKEND_CNT runs it will
#     collect metrics for one unmeasured backend. On the last run metrics for data transfer
PROFILING_RUN_CNT=$((BACKEND_CNT+1))
TEST_DRIVER_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"
ARTIFACT_PATH="$TEST_DRIVER_DIR/../.."
BENCHMARK_DRIVER_BIN=$ARTIFACT_PATH/Product/out/bin/tflite_run
REPORT_DIR=$ARTIFACT_PATH/report
RUN_TEST_SH=$ARTIFACT_PATH/tests/scripts/models/run_test.sh
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

    LOG_FILE=$REPORT_MODEL_DIR/tflite_${EXECUTOR,,}_$BACKEND.txt
    export OP_BACKEND_ALLOPS=$BACKEND
    export EXECUTOR=$EXECUTOR

    print_with_dots "$EXECUTOR $BACKEND without scheduler"

    RESULT=$(get_result_of_benchmark_test $BENCHMARK_DRIVER_BIN $MODEL $LOG_FILE)

    printf -v RESULT_INT '%d' $RESULT 2>/dev/null
    PERCENTAGE=$((100-RESULT_SCH_INT*100/RESULT_INT))
    echo "$RESULT ms. Parallel scheduler is $PERCENTAGE% faster"
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

        printf -v RESULT_SCH_INT '%d' $RESULT 2>/dev/null

        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_parallel.dot"

##################################################################################
        # Run Linear executor with scheduler
##################################################################################
        LOG_FILE=$REPORT_MODEL_DIR/tflite_linear_with_scheduler.txt
        export EXECUTOR="Linear"
        export GRAPH_DOT_DUMP=1
        print_with_dots "Linear with scheduler"

        RESULT=$(get_result_of_benchmark_test $BENCHMARK_DRIVER_BIN $MODEL $LOG_FILE)

        printf -v RESULT_INT '%d' $RESULT 2>/dev/null
        PERCENTAGE=$((100-RESULT_SCH_INT*100/RESULT_INT))
        echo "$RESULT ms. Parallel scheduler is $PERCENTAGE% faster"

        # Remove metrics so that for next model in profiler can get metrics
        #   for operations with input&output sizes the same as the model
        mv "exec_time.json" $REPORT_MODEL_DIR
        # Save the dot graph
        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_linear.dot"

##################################################################################
        # Turn off scheduler
##################################################################################
        export USE_SCHEDULER=0

        # Run LinearExecutor on acl_cl without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Linear" "acl_cl"
        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_linear_acl_cl.dot"

        # Run LinearExecutor on acl_neon without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Linear" "acl_neon"
        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_linear_acl_neon.dot"

        # Run ParallelExecutor on acl_cl without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Parallel" "acl_cl"
        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_parallel_acl_cl.dot"

        # Run ParallelExecutor on acl_neon without scheduler
        run_without_sched $RESULT_SCH_INT $REPORT_MODEL_DIR $MODEL "Parallel" "acl_neon"
        mv "after_lower_subg-0.dot" $REPORT_MODEL_DIR/"after_lower_subg-0_parallel_acl_neon.dot"

        unset GRAPH_DOT_DUMP

        if command -v dot;
        then
            dot -Tpng $REPORT_MODEL_DIR/"after_lower_subg-0_parallel.dot" -o $REPORT_MODEL_DIR/"parallel.png"
            dot -Tpng $REPORT_MODEL_DIR/"after_lower_subg-0_linear.dot" -o $REPORT_MODEL_DIR/"linear.png"
            dot -Tpng $REPORT_MODEL_DIR/"after_lower_subg-0_linear_acl_cl.dot" -o $REPORT_MODEL_DIR/"linear_acl_cl.png"
            dot -Tpng $REPORT_MODEL_DIR/"after_lower_subg-0_linear_acl_neon.dot" -o $REPORT_MODEL_DIR/"linear_acl_neon.png"
            dot -Tpng $REPORT_MODEL_DIR/"after_lower_subg-0_parallel_acl_cl.dot" -o $REPORT_MODEL_DIR/"paralle_acl_cl.png"
            dot -Tpng $REPORT_MODEL_DIR/"after_lower_subg-0_parallel_acl_neon.dot" -o $REPORT_MODEL_DIR/"parallel_acl_neon.png"
        fi

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
