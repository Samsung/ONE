#!/usr/bin/env bash
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
NNFW_HOME="$(dirname $(dirname $(dirname ${MY_PATH})))"
CACHE_ROOT_PATH=$MY_PATH/"cache"
TEST_ROOT_PATH=$MY_PATH/"config"
REPORT_DIR="report"

RUN_DISABLED="true"

function Usage()
{
    echo "Usage: ./$0 --driverbin={such as tflite_run} {tests to test or empty for all of tests}"
    echo "Usage: ./$0 --driverbin=Product/out/bin/tflite_run --reportdir=report --tapname=verification.tap avgpool1 avgpool2"
    echo ""
    echo "--download            - (default=on) Download model files"
    echo "--run                 - (default=on) Test model files"
    echo "--driverbin           - (default=../../Product/out/bin/tflite_run) Runner for runnning model tests"
    echo "--reportdir           - (default=report) Directory to place tap files"
    echo "--tapname             - (default=framework_test.tap) File name to be written for tap"
    echo "--md5                 - (default=on) MD5 check when download model files"
    echo ""
}

function need_download()
{
    LOCAL_PATH=$1
    REMOTE_URL=$2
    if [ ! -e $LOCAL_PATH ]; then
        return 0;
    fi
    # Ignore checking md5 in cache
    # TODO Use "--md5" option only and remove IGNORE_MD5 environment variable
    if [ ! -z $IGNORE_MD5 ] && [ "$IGNORE_MD5" == "1" ]; then
        return 1
    fi
    if [ "$MD5_CHECK" = "off" ]; then
        return 1
    fi

    LOCAL_HASH=$(md5sum $LOCAL_PATH | awk '{ print $1 }')
    REMOTE_HASH=$(curl -ss $REMOTE_URL | md5sum  | awk '{ print $1 }')
    # TODO Emit an error when Content-MD5 field was not found. (Server configuration issue)
    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
        echo "Downloaded file is outdated or incomplete."
        return 0
    fi
    return 1
}

DRIVER_BIN=""
TAP_NAME="framework_test.tap"
TEST_LIST=()
DOWNLOAD_MODEL="on"
RUN_TEST="on"
MD5_CHECK="on"

# Support environment variable setting for mirror server
FIXED_MODELFILE_SERVER="${MODELFILE_SERVER:-}"

for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --driverbin=*)
            DRIVER_BIN=${i#*=}
            ;;
        --reportdir=*)
            REPORT_DIR=${i#*=}
            ;;
        --tapname=*)
            TAP_NAME=${i#*=}
            ;;
        --download=*)
            DOWNLOAD_MODE=${i#*=}
            ;;
        --md5=*)
            MD5_CHECK=${i#*=}
            ;;
        --run=*)
            RUN_TEST=${i#*=}
            ;;
        *)
            TEST_LIST+=( $i )
            ;;
    esac
    shift
done

if [[ ${#TEST_LIST[@]} -eq 0 ]]; then
    RUN_DISABLED="false"
fi

if [ ! -n "$DRIVER_BIN" ]; then
    DRIVER_BIN="$NNFW_HOME/Product/out/bin/tflite_run"
fi

# Check test driver setting
if [ ! -e $DRIVER_BIN ] && [ "$RUN_TEST" = "on" ]; then
    echo "Cannot find test driver" $DRIVER_BIN ": please set proper DRIVER_BIN"
    exit 1
fi

run_tests()
{
    echo "1..$#" > $REPORT_DIR/$TAP_NAME
    SELECTED_TESTS=$@

    echo ""
    echo "Running tests:"
    echo "======================"
    for TEST_NAME in $SELECTED_TESTS; do
        echo $TEST_NAME
    done
    echo "======================"

    TOTAL_RESULT=0  # 0(normal) or 1(abnormal)
    i=0
    for TEST_NAME in $SELECTED_TESTS; do
        # Test configure initialization
        ((i++))
        STATUS="enabled"
        MODELFILE_SERVER_PATH=""
        MODELFILE_NAME=""
        source $TEST_ROOT_PATH/$TEST_NAME/config.sh

        LOWER_STATUS="$(echo $STATUS | awk '{print tolower($0)}')"
        if [ "$LOWER_STATUS" == "disabled" ] && [ "$RUN_DISABLED" == "false" ]; then
            echo ""
            echo "Skip $TEST_NAME"
            echo "======================"
            echo "ok $i # skip $TEST_NAME" >> $REPORT_DIR/$TAP_NAME
            continue
        fi

        TEST_CACHE_PATH=$CACHE_ROOT_PATH/$TEST_NAME
        MODELFILE=$TEST_CACHE_PATH/$MODELFILE_NAME

        # Find model file for downloaded by zip
        if [ "${MODELFILE_NAME##*.}" = "zip" ]; then
            pushd $TEST_CACHE_PATH
            MODELFILE=$TEST_CACHE_PATH/$(ls *.tflite)
            popd
        fi

        echo ""
        echo "Run $TEST_NAME"
        echo "======================"

        # Run driver to test framework
        $DRIVER_BIN $MODELFILE

        if [[ $? -eq 0 ]]; then
            echo "ok $i - $TEST_NAME" >> $REPORT_DIR/$TAP_NAME
        else
            echo "not ok $i - $TEST_NAME" >> $REPORT_DIR/$TAP_NAME
            TOTAL_RESULT=1
        fi
    done
    return $TOTAL_RESULT
}

download_tests()
{
    SELECTED_TESTS=$@

    echo ""
    echo "Downloading tests:"
    echo "======================"
    for TEST_NAME in $SELECTED_TESTS; do
        echo $TEST_NAME
    done
    echo "======================"

    i=0
    for TEST_NAME in $SELECTED_TESTS; do
        # Test configure initialization
        ((i++))
        MODELFILE_SERVER_PATH=""
        MODELFILE_NAME=""
        source $TEST_ROOT_PATH/$TEST_NAME/config.sh

        TEST_CACHE_PATH=$CACHE_ROOT_PATH/$TEST_NAME
        MODELFILE=$TEST_CACHE_PATH/$MODELFILE_NAME
        MODELFILE_URL="$MODELFILE_SERVER/$MODELFILE_NAME"
        if [ -n  "$FIXED_MODELFILE_SERVER" ]; then
            MODELFILE_URL="$FIXED_MODELFILE_SERVER/$MODELFILE_NAME"
        fi

        # Download model file
        if [ ! -e $TEST_CACHE_PATH ]; then
            mkdir -p $TEST_CACHE_PATH
        fi

        # Download unless we have it in cache (Also check md5sum)
        if need_download "$MODELFILE" "$MODELFILE_URL"; then
            echo ""
            echo "Download test file for $TEST_NAME"
            echo "======================"

            rm -f $MODELFILE # Remove invalid file if exists
            pushd $TEST_CACHE_PATH
            wget -nv $MODELFILE_URL
            if [ "${MODELFILE_NAME##*.}" == "zip" ]; then
                unzip -o $MODELFILE_NAME
            fi
            popd
        fi

    done
}


find_tests()
{
    local TEST_DIRS="$@"
    local TESTS_TO_RUN=""

    if [[ $# -eq 0 ]]; then
        TEST_DIRS="."
    fi

    shift $#

    pushd $TEST_ROOT_PATH > /dev/null
    for DIR in $TEST_DIRS; do
        if [ -d "$DIR" ]; then
            TESTS_FOUND=$(find "$DIR" -type f -name 'config.sh' -exec dirname {} \;| sed 's|^./||' | sort)
            TESTS_TO_RUN="$TESTS_TO_RUN $TESTS_FOUND"
        else
            echo "Test $DIR was not found. This test is not added." 1>&2
        fi
    done
    popd > /dev/null

    echo $TESTS_TO_RUN
}

mkdir -p $REPORT_DIR
TESTS_TO_RUN=$(find_tests ${TEST_LIST[@]})

if [ "$DOWNLOAD_MODEL" = "on" ]; then
    download_tests $TESTS_TO_RUN
fi

if [ "$RUN_TEST" = "on" ]; then
    run_tests $TESTS_TO_RUN
fi
exit $?
