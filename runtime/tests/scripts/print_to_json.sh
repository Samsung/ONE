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

JSON_BENCHMARK_REPORT_DIR=  # $ARTIFACT/report/benchmark
JSON_PRINT_TO_DIR=          # $ARTIFACT/report
JSON_RESULT_JSON=           # $ARTIFACT/report/benchmark_result.json or benchmark_op_result.json
JSON_MODELS_FILE_DIR=       # $ARTIFACT/report/benchmark

function echo_to_file
{
    echo -e "$1" >> $JSON_RESULT_JSON
}

function print_comma() # ,
{
    echo_to_file ","
}

function print_brace_start() # {
{
    echo_to_file "{"
}

function print_brace_end() # }
{
    echo_to_file "}"
}

function print_bracket_start() # "$NAME": [
{
    local NAME=$1
    echo_to_file "\"$NAME\":["
}

function print_bracket_end() # ]
{
    echo_to_file "]"
}

function print_key_value() # "$KEY": "$VALUE"
{
    local KEY=$1
    local VALUE=$2
    echo_to_file "\"$KEY\": \"$VALUE\""
}

function print_key_value_no_quot() # "$KEY": $VALUE
{
    local KEY=$1
    local VALUE=$2
    echo_to_file "\"$KEY\": $VALUE"
}

function print_key_value_dbl() # "dblValue": $VALUE
{
    local VALUE=$1
    echo_to_file "\"dblValue\": $VALUE"  # value should not include ""
}

function print_results()
{
    local NAME=$1
    local MEAN=$2

    print_bracket_start "results"
    print_brace_start
    print_key_value "name" "Mean_of_$NAME"
    print_comma
    print_key_value "unit" "ms"
    print_comma
    print_key_value_dbl "$MEAN"
    print_brace_end
    print_bracket_end
}

function print_test()
{
    local NAME=$1
    local RESULT=$2

    print_brace_start
    print_key_value "name" "$NAME"
    print_comma
    print_results "$NAME" "$RESULT"
    print_brace_end
}

function print_tests()
{
    local MODEL=$1
    local REPORT_MODEL_DIR=$JSON_BENCHMARK_REPORT_DIR/$MODEL
    local TEST_RESULTS=$(find $REPORT_MODEL_DIR -name "*.result" -exec basename {} \;)
    local TEST_NUM=$(find $REPORT_MODEL_DIR -name "*.result" | wc -l)

    print_bracket_start "tests"

    local i=0
    for TEST in $TEST_RESULTS; do
        local NAME=$(cat $REPORT_MODEL_DIR/$TEST | awk '{print $1}')
        local RESULT=$(cat $REPORT_MODEL_DIR/$TEST | awk '{print $2}')
        print_test $NAME $RESULT
        if [[ $i -ne $TEST_NUM-1 ]]; then
            print_comma
        fi
        i=$((i+1))
    done

    print_bracket_end
}

function print_groups()
{
    local TOTAL_MODEL_NUM=0
    local TOTAL_MODELS=

    for MODELS_FILE in $(find $JSON_MODELS_FILE_DIR -name "benchmark*_models.txt"); do
        # In $MODELS_FILE, there are only unique(not duplicated) model names.
        local MODEL_NUM=$(cat $MODELS_FILE | wc -l)
        TOTAL_MODEL_NUM=$((TOTAL_MODEL_NUM+MODEL_NUM))
        for MODELS in $(cat $MODELS_FILE); do
            TOTAL_MODELS+="$MODELS "
        done
    done

    print_bracket_start "groups"

    local i=0
    for MODEL in $TOTAL_MODELS; do
        print_brace_start
        print_key_value "name" " $MODEL"
        print_comma
        print_tests $MODEL
        print_brace_end
        if [[ $i -ne $TOTAL_MODEL_NUM-1 ]]; then
            print_comma
        fi
        i=$((i+1))
    done

    print_bracket_end
}

function print_to_json()
{
    JSON_BENCHMARK_REPORT_DIR=$1
    JSON_PRINT_TO_DIR=$2
    JSON_PRINT_TO_FILENAME=$3

    JSON_RESULT_JSON=$JSON_PRINT_TO_DIR/$JSON_PRINT_TO_FILENAME
    rm -f $JSON_RESULT_JSON
    JSON_MODELS_FILE_DIR=$JSON_BENCHMARK_REPORT_DIR

    print_brace_start
    print_groups
    print_brace_end
}
