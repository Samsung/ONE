#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

TEST_ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')
TEST_OS="android"
TEST_PLATFORM="$TEST_ARCH-$TEST_OS"
EXECUTORS=("Linear" "Dataflow" "Parallel")

# Model download server setting
if [[ -z "${MODELFILE_SERVER}" ]]; then
  echo "[ERROR] Model file server is not set"
  echo "        Need to download model file for test"
  exit 1
else
  echo "Model Server: ${MODELFILE_SERVER}"
fi

apt-get update && apt-get install -y curl

BACKENDS=( "acl_neon" "cpu" "acl_cl" )

$ROOT_PATH/tests/scripts/models/run_test.sh --download=on --run=off
$ROOT_PATH/Product/aarch64-android.release/out/test/models/run_test.sh --download=on --run=off \
  --configdir=$ROOT_PATH/Product/aarch64-android.release/out/test/models/nnfw_api_gtest \
  --cachedir=$ROOT_PATH/Product/aarch64-android.release/out/unittest_standalone/nnfw_api_gtest_models

N=`adb devices 2>/dev/null | wc -l`

# exit if no device found
if [[ $N -le 2 ]]; then
    echo "No device found."
    exit 1;
fi

NUM_DEV=$(($N-2))
echo "device list"
DEVICE_LIST=`adb devices 2>/dev/null`
echo "$DEVICE_LIST" | tail -n"$NUM_DEV"

if [ -z "$SERIAL" ]; then
    SERIAL=`echo "$DEVICE_LIST" | tail -n1 | awk '{print $1}'`
fi
ADB_CMD="adb -s $SERIAL "

# root on, remount as rw
$ADB_CMD root on
$ADB_CMD shell mount -o rw,remount /

$ADB_CMD shell rm -rf /data/local/tmp/onert_android
$ADB_CMD shell mkdir -p /data/local/tmp/onert_android/report
$ADB_CMD push $ROOT_PATH/tests /data/local/tmp/onert_android/.
$ADB_CMD push $ROOT_PATH/Product/aarch64-android.release/out /data/local/tmp/onert_android/Product/.

$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib BACKEND=acl_cl sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/onert_android/Product/bin/tflite_loader_test_tool \
                                                        --reportdir=/data/local/tmp/onert_android/report \
                                                        --tapname=tflite_loader.tap
for BACKEND in "${BACKENDS[@]}";
do
for EXECUTOR in "${EXECUTORS[@]}";
do
MODELLIST=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.${TEST_ARCH}.${BACKEND}.txt")
# $ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib EXECUTOR=$EXECUTOR BACKEND=$BACKEND sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
#                                                         --driverbin=/data/local/tmp/onert_android/Product/unittest/nnapi_gtest \
#                                                         --reportdir=/data/local/tmp/onert_android/report \
#                                                         --tapname=nnapi_gtest_$BACKEND_$EXECUTOR.tap
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib EXECUTOR=$EXECUTOR BACKEND=$BACKEND sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/onert_android/Product/bin/nnapi_test \
                                                        --reportdir=/data/local/tmp/onert_android/report \
                                                        --tapname=nnapi_test_$BACKEND_$EXECUTOR.tap
done
done
MODELLIST=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.noarch.interp.txt")
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib BACKEND="" DISABLE_COMPILE=1 sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/onert_android/Product/bin/nnapi_test \
                                                        --reportdir=/data/local/tmp/onert_android/report \
                                                        --tapname=nnapi_test_interp.tap ${MODELLIST:-}
$ADB_CMD shell LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib USE_NNAPI=1 sh /data/local/tmp/onert_android/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=/data/local/tmp/onert_android/Product/bin/tflite_run \
                                                        --reportdir=/data/local/tmp/onert_android/report \
                                                        --tapname=tflite_run.tap

# This is test for profiling.
# $ADB_CMD shell mkdir -p /data/local/tmp/onert_android/report/benchmark
# $ADB_CMD shell 'cd /data/local/tmp/onert_android && LD_LIBRARY_PATH=/data/local/tmp/onert_android/Product/lib sh /data/local/tmp/onert_android/tests/scripts/test_scheduler_with_profiling_android.sh'

mkdir -p $ROOT_PATH/report
rm -rf $ROOT_PATH/report/android

$ADB_CMD pull /data/local/tmp/onert_android/report $ROOT_PATH/report/android
