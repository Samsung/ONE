#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

: ${DEVICE:="none"}
TEST_ARCH="aarch64"
TEST_OS="android"
TEST_PLATFORM="$TEST_ARCH-$TEST_OS"
EXECUTORS=("Linear" "Dataflow" "Parallel")
BACKENDS=( "acl_neon" "cpu" "acl_cl" )
ANDROID_WORKDIR="/data/local/tmp/onert_android"
ANDROID_REPORT_DIR="$ANDROID_WORKDIR/report"

# Model download server setting
if [[ -z "${MODELFILE_SERVER}" ]]; then
  echo "[ERROR] Model file server is not set"
  echo "        Need to download model file for test"
  exit 1
else
  echo "Model Server: ${MODELFILE_SERVER}"
fi

apt-get update && apt-get install -y curl



# Download Model
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

$ADB_CMD shell rm -rf $ANDROID_WORKDIR
$ADB_CMD shell rm -rf /data/local/tmp/TestCompilationCaching*
$ADB_CMD shell mkdir -p $ANDROID_REPORT_DIR
$ADB_CMD push $ROOT_PATH/tests $ANDROID_WORKDIR/.
$ADB_CMD push $ROOT_PATH/Product/aarch64-android.release/out $ANDROID_WORKDIR/Product/.

#TFloader Testing
TESTLIST=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/tflite_loader_list.${TEST_ARCH}.txt")
$ADB_CMD shell LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib BACKENDS=acl_cl sh $ANDROID_WORKDIR/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=$ANDROID_WORKDIR/Product/bin/tflite_loader_test_tool \
                                                        --reportdir=$ANDROID_REPORT_DIR \
                                                        --tapname=tflite_loader.tap ${TESTLIST:-}
#Union SkipList and testing List Creation
UNION_MODELLIST_PREFIX="${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.${TEST_ARCH}"
UNION_SKIPLIST_PREFIX="${ROOT_PATH}/Product/aarch64-android.release/out/unittest/nnapi_gtest.skip.${TEST_PLATFORM}"
sort $UNION_MODELLIST_PREFIX.${BACKENDS[0]}.txt > $UNION_MODELLIST_PREFIX.intersect.txt
sort $UNION_SKIPLIST_PREFIX.${BACKENDS[0]} > $UNION_SKIPLIST_PREFIX.union
for BACKEND in "${BACKENDS[@]:1}"; do
  comm -12 <(sort $UNION_MODELLIST_PREFIX.intersect.txt) <(sort $UNION_MODELLIST_PREFIX.$BACKEND.txt) > $UNION_MODELLIST_PREFIX.intersect.next.txt
  comm <(sort $UNION_SKIPLIST_PREFIX.union) <(sort $UNION_SKIPLIST_PREFIX.$BACKEND) | tr -d "[:blank:]" > $UNION_SKIPLIST_PREFIX.union.next
  mv $UNION_MODELLIST_PREFIX.intersect.next.txt $UNION_MODELLIST_PREFIX.intersect.txt
  mv $UNION_SKIPLIST_PREFIX.union.next $UNION_SKIPLIST_PREFIX.union
done
# Fail on NCHW layout (acl_cl, acl_neon)
# TODO Fix bug
echo "GeneratedTests.*weights_as_inputs*" >> $UNION_SKIPLIST_PREFIX.union
echo "GeneratedTests.logical_or_broadcast_4D_2D_nnfw" >> $UNION_SKIPLIST_PREFIX.union
echo "GeneratedTests.mean" >> $UNION_SKIPLIST_PREFIX.union
echo "GeneratedTests.add_broadcast_4D_2D_after_nops_float_nnfw" >> $UNION_SKIPLIST_PREFIX.union
echo "GeneratedTests.argmax_*" >> $UNION_SKIPLIST_PREFIX.union
echo "GeneratedTests.squeeze_relaxed" >> $UNION_SKIPLIST_PREFIX.union

# Testing Each BACKEND with EXECUTOR Combination
for BACKEND in "${BACKENDS[@]}"; do
  for EXECUTOR in "${EXECUTORS[@]}"; do
    MODELLIST=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.${TEST_ARCH}.${BACKEND}.txt")
    SKIPLIST=$(grep -v '#' "${ROOT_PATH}/Product/aarch64-android.release/out/unittest/nnapi_gtest.skip.${TEST_PLATFORM}.${BACKEND}" | tr '\n' ':')
    $ADB_CMD shell LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib EXECUTOR=$EXECUTOR BACKENDS=$BACKEND $ANDROID_WORKDIR/Product/unittest/nnapi_gtest \
                                                            --gtest_output=xml:$ANDROID_REPORT_DIR/nnapi_gtest_${BACKEND}_${EXECUTOR}.xml \
                                                            --gtest_filter=-${SKIPLIST}
    $ADB_CMD shell LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib EXECUTOR=$EXECUTOR BACKENDS=$BACKEND sh $ANDROID_WORKDIR/tests/scripts/models/run_test_android.sh \
                                                            --driverbin=$ANDROID_WORKDIR/Product/bin/nnapi_test \
                                                            --reportdir=$ANDROID_REPORT_DIR \
                                                            --tapname=nnapi_test_${BACKEND}_${EXECUTOR}.tap ${MODELLIST:-}
  done
done

# Testing Interperter
MODELLIST_INTERP=$(cat "${ROOT_PATH}/Product/aarch64-android.release/out/test/list/frameworktest_list.noarch.interp.txt")
SKIPLIST_INTERP=$(grep -v '#' "${ROOT_PATH}/Product/aarch64-android.release/out/unittest/nnapi_gtest.skip.noarch.interp" | tr '\n' ':')
$ADB_CMD shell LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib EXECUTOR=Interpreter BACKENDS=$BACKEND $ANDROID_WORKDIR/Product/unittest/nnapi_gtest \
                                                            --gtest_output=xml:$ANDROID_REPORT_DIR/nnapi_gtest_interp_Interpreter.xml \
                                                            --gtest_filter=-$SKIPLIST_INTERP
$ADB_CMD shell LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib EXECUTOR=Interpreter BACKENDS="" DISABLE_COMPILE=1 sh $ANDROID_WORKDIR/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=$ANDROID_WORKDIR/Product/bin/nnapi_test \
                                                        --reportdir=$ANDROID_REPORT_DIR \
                                                        --tapname=nnapi_test_interp_Interpreter.tap ${MODELLIST_INTERP:-}

# Testing Mixed Backend
MODELLIST_UNION=$(cat "${UNION_MODELLIST_PREFIX}.intersect.txt")
SKIPLIST_UNION=$(grep -v '#' "$UNION_SKIPLIST_PREFIX.union" | tr '\n' ':')
$ADB_CMD shell LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib OP_BACKEND_Conv2D="cpu" OP_BACKEND_MaxPool2D="acl_cl" OP_BACKEND_AvgPool2D="acl_neon" ACL_LAYOUT="NCHW" BACKENDS="acl_cl\;acl_neon\;cpu" $ANDROID_WORKDIR/Product/unittest/nnapi_gtest \
                                                            --gtest_output=xml:$ANDROID_REPORT_DIR/nnapi_gtest_mixed.xml \
                                                            --gtest_filter=-${SKIPLIST_UNION}
$ADB_CMD shell LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib OP_BACKEND_Conv2D="cpu" OP_BACKEND_MaxPool2D="acl_cl" OP_BACKEND_AvgPool2D="acl_neon" ACL_LAYOUT="NCHW" BACKENDS="acl_cl\;acl_neon\;cpu" sh $ANDROID_WORKDIR/tests/scripts/models/run_test_android.sh \
                                                        --driverbin=$ANDROID_WORKDIR/Product/bin/nnapi_test \
                                                        --reportdir=$ANDROID_REPORT_DIR \
                                                        --tapname=nnapi_test_mixed.tap ${MODELLIST_UNION:-}

# Testing Unittest_standalone
for TEST_BIN in `find ${ROOT_PATH}/Product/aarch64-android.release/out/unittest_standalone -maxdepth 1 -type f -executable`; do
  $ADB_CMD shell "cd $ANDROID_WORKDIR/Product/unittest_standalone;
                  LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib ./$(basename $TEST_BIN) \
                  --gtest_output=xml:$ANDROID_REPORT_DIR/$(basename $TEST_BIN).xml"
done

# This is test for profiling.
# $ADB_CMD shell mkdir -p $ANDROID_REPORT_DIR/benchmark
# $ADB_CMD shell 'cd $ANDROID_WORKDIR && LD_LIBRARY_PATH=$ANDROID_WORKDIR/Product/lib sh $ANDROID_WORKDIR/tests/scripts/test_scheduler_with_profiling_android.sh'

rm -rf $ROOT_PATH/report
mkdir -p $ROOT_PATH/report


$ADB_CMD pull $ANDROID_REPORT_DIR $ROOT_PATH/report/$DEVICE
