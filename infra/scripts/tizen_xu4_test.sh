#!/bin/bash

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOST_HOME=$SCRIPT_ROOT/../..
if [ -z "$TEST_ROOT" ]; then
    TEST_ROOT=/opt/usr/nnfw-test
fi

function Usage()
{
    echo "Usage: ./tizen_xu4_test.sh --rpm-dir=path/to/rpm-dir --unittest --verification"
    echo "Usage: ./tizen_xu4_test.sh --test-suite-path=path/to/test-suite.tar.gz --unittest --verification"
    echo "--rpm-dir : directory containing nnfw.rpm and nnfw-test.rpm"
    echo "--test-suite-path : filepath to test-suite.tar.gz"
    echo "--gcov-dir : directory to save gcov files"
}


function prepare_rpm_test()
{
    echo "======= Test with rpm packages(gbs build) ======="
    # clean up
    $SDB_CMD shell rm -rf $TEST_ROOT
    $SDB_CMD shell mkdir -p $TEST_ROOT
    # install nnfw nnfw-test rpms
    for file in $RPM_DIR/*.rpm
    do
        $SDB_CMD push $file $TEST_ROOT
        $SDB_CMD shell rpm -Uvh $TEST_ROOT/$(basename $file) --force --nodeps
    done

    # download tflite model files
    pushd $HOST_HOME
    tests/scripts/framework/run_test.sh --download=on
    find tests -name "*.zip" -exec rm {} \;
    tar -zcf cache.tar.gz tests/scripts/framework/cache
    $SDB_CMD push cache.tar.gz $TEST_ROOT/.
    rm -rf cache.tar.gz
    $SDB_CMD shell tar -zxf $TEST_ROOT/cache.tar.gz -C $TEST_ROOT
}

function prepare_suite_test()
{
    echo "======= Test with test-suite(cross build) ======="
    # clean up
    $SDB_CMD shell rm -rf $TEST_ROOT
    $SDB_CMD shell mkdir -p $TEST_ROOT

    # install test-suite
    $SDB_CMD push $TEST_SUITE_PATH $TEST_ROOT/$(basename $TEST_SUITE_PATH)
    $SDB_CMD shell tar -zxf $TEST_ROOT/$(basename $TEST_SUITE_PATH) -C $TEST_ROOT

    # download tflite model files
    pushd $HOST_HOME
    tests/scripts/framework/run_test.sh --download=on
    find tests -name "*.zip" -exec rm {} \;
    tar -zcf cache.tar.gz tests/scripts/framework/cache
    $SDB_CMD push cache.tar.gz $TEST_ROOT/.
    rm -rf cache.tar.gz
    $SDB_CMD shell tar -zxf $TEST_ROOT/cache.tar.gz -C $TEST_ROOT
}


# Parse command argv
for i in "$@"
do
    case $i in
        -h|--help|help)
            Usage
            exit 1
            ;;
        --rpm-dir=*)
            RPM_DIR=${i#*=}
            ;;
        --test-suite-path=*)
            TEST_SUITE_PATH=${i#*=}
            ;;
        --gcov-dir=*)
            GCOV_DIR=${i#*=}
            ;;
    esac
    shift
done


N=`sdb devices 2>/dev/null | wc -l`

# exit if no device found
if [[ $N -le 1 ]]; then
    echo "No device found."
    exit 1;
fi

NUM_DEV=$(($N-1))
echo "device list"
DEVICE_LIST=`sdb devices 2>/dev/null`
echo "$DEVICE_LIST" | tail -n"$NUM_DEV"

if [ -z "$SERIAL" ]; then
    SERIAL=`echo "$DEVICE_LIST" | tail -n1 | awk '{print $1}'`
fi
SDB_CMD="sdb -s $SERIAL "

# root on, remount as rw
$SDB_CMD root on
$SDB_CMD shell mount -o rw,remount /

SCRIPT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$SCRIPT_ROOT/../

if [ -z "$RPM_DIR" ] && [ -z "$TEST_SUITE_PATH" ]; then
    echo "Please provide --rpm-dir or --test-suite-path"
    exit 255
fi

if [ ! -z "$RPM_DIR" ]; then
    prepare_rpm_test
else
    prepare_suite_test
fi

if [ -z "${GCOV_DIR}" ]; then
  ${SDB_CMD} shell /bin/bash -c "IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime.sh --backend acl_cl --tflite-loader"
  ${SDB_CMD} shell /bin/bash -c "IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime.sh --backend acl_neon"
  ${SDB_CMD} shell /bin/bash -c "IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime.sh --backend cpu"
  ${SDB_CMD} shell /bin/bash -c "IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime_mixed.sh"
  ${SDB_CMD} shell /bin/bash -c "IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime_interp.sh"
else
  mkdir -p ${GCOV_DIR}
  rm -rf ${GCOV_DIR}/*
  pushd ${GCOV_DIR}

  sdb pull ${TEST_ROOT}/tests/scripts/build_path.txt
  SRC_PREFIX=`cat build_path.txt`
  GCOV_PREFIX_STRIP=`echo "${SRC_PREFIX}" | grep -o '/' | wc -l`
  GCOV_DATA_PATH="/opt/usr/nnfw-gcov"

  # TODO For coverage check, we run acl_cl and mixed test
  ${SDB_CMD} shell /bin/bash -c "GCOV_PREFIX_STRIP=${GCOV_PREFIX_STRIP} IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime.sh --backend acl_cl --tflite-loader"
  ${SDB_CMD} shell /bin/bash -c "GCOV_PREFIX_STRIP=${GCOV_PREFIX_STRIP} IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime.sh --backend acl_neon"
  ${SDB_CMD} shell /bin/bash -c "GCOV_PREFIX_STRIP=${GCOV_PREFIX_STRIP} IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime.sh --backend cpu"
  ${SDB_CMD} shell /bin/bash -c "GCOV_PREFIX_STRIP=${GCOV_PREFIX_STRIP} IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime_mixed.sh"
  ${SDB_CMD} shell /bin/bash -c "GCOV_PREFIX_STRIP=${GCOV_PREFIX_STRIP} IGNORE_MD5=1 ${TEST_ROOT}/infra/scripts/test_ubuntu_runtime_interp.sh"

  # More test to check coverage
  ${SDB_CMD} shell "rm -rf ${GCOV_DATA_PATH} && mkdir -p ${GCOV_DATA_PATH}"
  ${SDB_CMD} shell "find ${TEST_ROOT} -type f \( -iname '*.gcda' -or -iname '*.gcno' \) -exec cp {} ${GCOV_DATA_PATH}/. \;"
  ${SDB_CMD} shell "cd ${TEST_ROOT} && tar -zcvf coverage-data.tar.gz -C ${GCOV_DATA_PATH} ."

  # pull gcov files
  sdb pull ${TEST_ROOT}/coverage-data.tar.gz
  tar -zxvf coverage-data.tar.gz
  popd
fi
