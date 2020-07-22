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

FWTEST_DRIVER_BIN=
FWTEST_REPORT_DIR=
FWTEST_TAP_NAME=
FWTEST_LOG_NAME=
FWTEST_TEST_NAME=

function Usage()
{
    echo "Usage Example:"
    echo "./$0 \\"
    echo "  --driverbin=Product/out/bin/tflite_run \\  # Test driver path"
    echo "  --frameworktest_list_file=tests/scripts/list/frameworktest_list.armv7l.cpu.txt \\"
    echo "  --reportdir=report \\            # Directory for the report files will be saved"
    echo "  --tapname=framework_test.tap \\  # Tap file name"
    echo "  --logname=framework_test.log \\  # Log file name"
    echo "  --testname=Frameworktest         # Name of the test just a label of tests"

    exit 1
}

for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            ;;
        --driverbin=*)
            FWTEST_DRIVER_BIN=${i#*=}
            ;;
        --reportdir=*)
            FWTEST_REPORT_DIR=${i#*=}
            ;;
        --tapname=*)
            FWTEST_TAP_NAME=${i#*=}
            ;;
        --logname=*)
            FWTEST_LOG_NAME=${i#*=}
            ;;
        --testname=*)
            FWTEST_TEST_NAME=${i#*=}
            ;;
        --frameworktest_list_file=*)
            FRAMEWORKTEST_LIST_FILE=${i#*=}
            ;;
    esac
    shift
done

[ ! -z "$FWTEST_DRIVER_BIN" ] || Usage
[ ! -z "$FWTEST_REPORT_DIR" ] || Usage
[ ! -z "$FWTEST_TAP_NAME" ] || Usage
[ ! -z "$FWTEST_LOG_NAME" ] || Usage
[ ! -z "$FWTEST_TEST_NAME" ] || Usage

if [ ! -e "$FWTEST_REPORT_DIR" ]; then
    mkdir -p $FWTEST_REPORT_DIR
fi

echo ""
echo "============================================"
echo "$FWTEST_TEST_NAME with $(basename $FWTEST_DRIVER_BIN) ..."

if [ ! -z "$FRAMEWORKTEST_LIST_FILE" ]; then
    MODELLIST=$(cat "${FRAMEWORKTEST_LIST_FILE}")
fi

$MY_PATH/framework/run_test.sh --driverbin=$FWTEST_DRIVER_BIN \
    --reportdir=$FWTEST_REPORT_DIR \
    --tapname=$FWTEST_TAP_NAME \
    ${MODELLIST:-} \
    > $FWTEST_REPORT_DIR/$FWTEST_LOG_NAME 2>&1
FWTEST_RESULT=$?
if [[ $FWTEST_RESULT -ne 0 ]]; then
    echo ""
    cat $FWTEST_REPORT_DIR/$FWTEST_TAP_NAME
    echo ""
    echo "$FWTEST_TEST_NAME failed... exit code: $FWTEST_RESULT"
    echo "============================================"
    echo ""
    exit $FWTEST_RESULT
fi

echo ""
cat $FWTEST_REPORT_DIR/$FWTEST_TAP_NAME
echo "============================================"
echo ""
