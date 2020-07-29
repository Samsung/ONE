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

UNITTEST_REPORT_DIR=
UNITTEST_TEST_DIR=
UNITTEST_RESULT=0
UNITTEST_RUN_ALL=""

function Usage()
{
    # TODO: Fill this
    echo "Usage: LD_LIBRARY_PATH=Product/out/lib ./$0 --reportdir=report --unittestdir=Product/out/unittest"
}

get_gtest_option()
{
    local UNITTEST_REPORT_FILE=$(basename $TEST_BIN)
    local output_option="--gtest_output=xml:$UNITTEST_REPORT_DIR/$UNITTEST_REPORT_FILE.xml"
    local filter_option
    if [ -r "$TEST_BIN.skip" ]; then
      filter_option="--gtest_filter=-$(grep -v '#' "$TEST_BIN.skip" | tr '\n' ':')"
    fi
    echo "$output_option $filter_option"
}

for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --reportdir=*)
            UNITTEST_REPORT_DIR=${i#*=}
            ;;
        --unittestdir=*)
            UNITTEST_TEST_DIR=${i#*=}
            ;;
        --runall)
            UNITTEST_RUN_ALL="true"
    esac
    shift
done

# TODO: handle exceptions for params

if [ ! -e "$UNITTEST_REPORT_DIR" ]; then
    mkdir -p $UNITTEST_REPORT_DIR
fi

echo ""
echo "============================================"
echo "Unittest start"
echo "============================================"

num_unittest=0
# Run all executables in unit test directory
for TEST_BIN in `find $UNITTEST_TEST_DIR -maxdepth 1 -type f -executable`; do
    num_unittest=$((num_unittest+1))
    echo "============================================"
    echo "Starting set $num_unittest: $TEST_BIN..."
    echo "============================================"
    TEMP_UNITTEST_RESULT=0

    if [ "$UNITTEST_RUN_ALL" == "true" ]; then
        for TEST_LIST_VERBOSE_LINE in $($TEST_BIN --gtest_list_tests); do
            if [[ $TEST_LIST_VERBOSE_LINE == *\. ]]; then
                TEST_LIST_CATEGORY=$TEST_LIST_VERBOSE_LINE
            else
                TEST_LIST_ITEM="$TEST_LIST_CATEGORY""$TEST_LIST_VERBOSE_LINE"
                $TEST_BIN --gtest_filter=$TEST_LIST_ITEM --gtest_output="xml:$UNITTEST_REPORT_DIR/$TEST_LIST_ITEM.xml"
            fi
        done
    else
        $TEST_BIN $(get_gtest_option)
        TEMP_UNITTEST_RESULT=$?
    fi

    if [[ $TEMP_UNITTEST_RESULT -ne 0 ]]; then
        UNITTEST_RESULT=$TEMP_UNITTEST_RESULT
        echo "$TEST_BIN failed... return code: $TEMP_UNITTEST_RESULT"
    fi
    echo "============================================"
    echo "Finishing set $num_unittest: $TEST_BIN..."
    echo "============================================"
done

if [[ $UNITTEST_RESULT -ne 0 ]]; then
    echo "============================================"
    echo "Failed unit test... exit code: $UNITTEST_RESULT"
    echo "============================================"
    exit $UNITTEST_RESULT
fi

echo "============================================"
echo "Completed total $num_unittest set of unittest"
echo "Unittest end"
echo "============================================"
