#!/bin/bash

# TODO : tizen sdb support
# TODO : multiple backend at once
#
# This benchmark tool works as follows:
# 0. Prepare test-suite
#
# On building, set make target to build_test_suite. This will create test-suite.tar.gz under Product/out directory.
# ```
# $ make build_test_suite
# ```
#
# 1. Install test-suite into target devices
#   - On android, test-suite should be located on /data/local/tmp/
#   - On Tizen, nnfw-test pacakge will install test-suite into /opt/usr/nnfw-test/
#
# 2. Prepare nnpackge
#
# 3. Run benchmark
#
# ```
# $./benchmark.sh --backend=cpu --num_runs=5 --nnpackge=/path/to/nnpkg
#
# ```
# 4. Result trace.json
#  - trace.json is the result file

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

## If no optional argument is passed, set for android
TEST_ROOT=/data/local/tmp/
BRIDGE=adb
BACKENDS=cpu
NUM_RUNS=3

function Usage()
{
    echo "Usage: ./benchamrk.sh --bridge=adb --backends=cpu --num_runs=5 --nnpackge=/path/to/nnpkg"
    echo ""
    echo "--bridge                  : adb or sdb"
    echo "--nnpackage=<dir>         : directory containing nnpackage"
    echo "--num_runs                : number of runs"
    echo "--backends                : backend list"
}

# Parse command argv
for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --bridge=*)
            BRIDGE=${i#*=}
            ;;
        --bridge)
            BRIDGE="$2"
            shift
            ;;
        --num_runs=*)
            NUM_RUNS=${i#*=}
            ;;
        --num_runs)
            NUM_RUNS="$2"
            shift
            ;;
        --nnpackage=*)
            NNPKG_PATH=${i#*=}
            ;;
        --nnpackage)
            NNPKG_PATH="$2"
            shift
            ;;
    esac
    shift
done


NNPKG_PATH_TARGET=$TEST_ROOT/nnpkg/`basename $NNPKG_PATH`

# 0. Push nnpackage into targeta
echo "Pusing nnpackge into ${NNPKG_PATH_TARGET}"
pushd $NNPKG_PATH/.. > /dev/null
tar -zcf nnpkg.tar.gz `basename $NNPKG_PATH`
$BRIDGE push nnpkg.tar.gz $TEST_ROOT
rm nnpkg.tar.gz
popd > /dev/null
$BRIDGE shell mkdir -p $TEST_ROOT/nnpkg
$BRIDGE shell tar -zxf $TEST_ROOT/nnpkg.tar.gz -C $TEST_ROOT/nnpkg
$BRIDGE shell rm $TEST_ROOT/nnpkg.tar.gz

# 1. Run
$BRIDGE shell LD_LIBRARY_PATH=$TEST_ROOT/Product/out/lib TRACING_MODE=1 WORKSPACE_DIR=$TEST_ROOT BACKENDS=$BACKENDS $TEST_ROOT/Product/out/bin/onert_run --nnpackage $NNPKG_PATH_TARGET -r $NUM_RUNS

# 2. Pull result file
echo "Pulling data from target to trace.json"
$BRIDGE pull $TEST_ROOT/trace.json

# 3. Clean up
$BRIDGE shell rm -rf $TEST_ROOT/nnpkg
