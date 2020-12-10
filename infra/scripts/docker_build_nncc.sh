#!/bin/bash

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
if [ -d $TENSORFLOW_PREFIX ]; then
  DOCKER_OPTS+=" -v $TENSORFLOW_PREFIX:/opt/tensorflow"
  CONFIG_OPTIONS+=" -DTENSORFLOW_PREFIX=/opt/tensorflow"
fi

# prepare onnx
if [ -d $ONNXRUNTIME_PREFIX ]; then
  DOCKER_OPTS+=" -v $ONNXRUNTIME_PREFIX:/opt/onnxruntime"
  CONFIG_OPTIONS+=" -DONNXRUNTIME_PREFIX=/opt/onnxruntime"
fi

# docker image name
# - for xenial, use DOCKER_IMAGE_NAME="nnfw/one-devtools:xenial"
# - for bionic, use DOCKER_IMAGE_NAME="nnfw/one-devtools:bionic"
if [[ -z $DOCKER_IMAGE_NAME ]]; then
  echo "It will use default docker image name"
fi

NNAS_WORKSPACE=${NNAS_WORKSPACE:-build}
NNCC_INSTALL_PREFIX=${NNAS_WORKSPACE}/out
DOCKER_OPTS+=" -e NNAS_BUILD_PREFIX=${NNAS_WORKSPACE}"
export DOCKER_OPTS
if [[ -z "${ARCHIVE_PATH}" ]]; then
  ARCHIVE_PATH=${NNAS_WORKSPACE}/archive
fi

set -e

pushd $ROOT_PATH > /dev/null

mkdir -p ${NNCC_INSTALL_PREFIX}
./nncc docker-run ./nnas create-package --prefix "${PWD}/${NNCC_INSTALL_PREFIX}" -- "${CONFIG_OPTIONS}"

mkdir -p ${ARCHIVE_PATH}
tar -zcf ${ARCHIVE_PATH}/nncc-package.tar.gz -C ${NNCC_INSTALL_PREFIX} --exclude test --exclude tflchef* ./
tar -zcf ${ARCHIVE_PATH}/nncc-test-package.tar.gz -C ${NNCC_INSTALL_PREFIX} ./test

popd > /dev/null
