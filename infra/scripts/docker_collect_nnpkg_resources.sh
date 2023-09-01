#!/bin/bash

function join_by
{
  local IFS="$1"; shift; echo "$*"
}

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

CONFIG_OPTIONS=""
# mount volume (or directory) for externals
if [ -n "$EXTERNAL_VOLUME" ]; then
  DOCKER_OPTS+=" -v $EXTERNAL_VOLUME:/externals"
  CONFIG_OPTIONS+=" -DNNAS_EXTERNALS_DIR=/externals"
else
  echo "It will use default external path"
fi

# mount volume (or directory) for overlay
if [ -n "$OVERLAY_VOLUME" ]; then
  DOCKER_OPTS+=" -v $OVERLAY_VOLUME:/overlay"
  CONFIG_OPTIONS+=" -DNNCC_OVERLAY_DIR=/overlay"
else
  echo "It will use default overlay path"
fi

# prepare tensorflow
if [ -n "$TENSORFLOW_PREFIX" ]; then
  DOCKER_OPTS+=" -v $TENSORFLOW_PREFIX:/opt/tensorflow"
  CONFIG_OPTIONS+=" -DTENSORFLOW_PREFIX=/opt/tensorflow"
fi

# prepare onnx
if [ -n "$ONNXRUNTIME_PREFIX" ]; then
  DOCKER_OPTS+=" -v $ONNXRUNTIME_PREFIX:/opt/onnxruntime"
  CONFIG_OPTIONS+=" -DONNXRUNTIME_PREFIX=/opt/onnxruntime"
fi

# docker image name
# - for bionic, use DOCKER_IMAGE_NAME="nnfw/one-devtools:bionic"
# - for focal, use DOCKER_IMAGE_NAME="nnfw/one-devtools:focal"
if [[ -z $DOCKER_IMAGE_NAME ]]; then
  echo "It will use default docker image name"
fi

# Assume that build is already finished, and ready to test
NNAS_WORKSPACE=${NNAS_WORKSPACE:-build}
export NNCC_WORKSPACE=${NNAS_WORKSPACE}/nncc
export DOCKER_OPTS

if [[ -z "${ARCHIVE_PATH}" ]]; then
  ARCHIVE_PATH=${NNAS_WORKSPACE}/archive
fi

set -e

pushd $ROOT_PATH > /dev/null

REQUIRED_UNITS=()
# Common Libraries
REQUIRED_UNITS+=("angkor" "cwrap" "pepper-str" "pepper-strcast" "pp")
REQUIRED_UNITS+=("pepper-csv2vec")
REQUIRED_UNITS+=("oops" "safemain" "foder" "crew" "arser" "vconone")
# Hermes Logging Framework
REQUIRED_UNITS+=("hermes" "hermes-std")
# loco IR and related utilities
REQUIRED_UNITS+=("loco" "locop" "locomotiv" "logo-core" "logo")
# Compute
REQUIRED_UNITS+=("luci-compute")
# Circle compiler library (.circle -> .circle)
REQUIRED_UNITS+=("luci")
# Flatbuffer I/O
REQUIRED_UNITS+=("mio-tflite280" "mio-circle05" "mio-tflite2121" "mio-circle06")
# Tools
REQUIRED_UNITS+=("tflite2circle" "circle2circle" "luci-interpreter")
REQUIRED_UNITS+=("souschef" "tflchef" "circlechef" "circle-verify")
# common-artifacts
REQUIRED_UNITS+=("common-artifacts")

# Reset whitelist to build all
./nncc docker-run ./nncc configure -DENABLE_STRICT_BUILD=ON -DCMAKE_BUILD_TYPE=release \
  -DBUILD_WHITELIST=$(join_by ";" "${REQUIRED_UNITS[@]}") \
  $CONFIG_OPTIONS
./nncc docker-run ./nncc build -j4

mkdir -p ${ARCHIVE_PATH}
TEMP_DIR=$(mktemp -d -t resXXXX)
rm -f ${TEMP_DIR}/*
mkdir -p ${TEMP_DIR}/nnpkg-tcs

# Copy nnpakcage only if it has its test data
for nnpkg in $NNCC_WORKSPACE/compiler/common-artifacts/*; do
  if [ -d $nnpkg/metadata/tc ]; then
    cp -r $nnpkg ${TEMP_DIR}/nnpkg-tcs
  fi
done

tar -zcf ${ARCHIVE_PATH}/nnpkg-test-suite.tar.gz -C ${TEMP_DIR} ./
rm -rf ${TEMP_DIR}

echo "resouce generation end"
popd > /dev/null
