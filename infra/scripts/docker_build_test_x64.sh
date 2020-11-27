#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

# mount volume (or directory) for externals
if [ -n "$EXTERNAL_VOLUME" ]; then
  DOCKER_VOLUMES+=" -v $EXTERNAL_VOLUME:/externals"
  DOCKER_ENV_VARS+=" -e EXTERNAL_VOLUME=/externals"
else
  echo "It will use default external path"
fi

# docker image name
# - for xenial, use DOCKER_IMAGE_NAME="nnfw/one-devtools:xenial"
# - for bionic, use DOCKER_IMAGE_NAME="nnfw/one-devtools:bionic"
if [[ -z $DOCKER_IMAGE_NAME ]]; then
  echo "It will use default docker image name"
fi

# Mirror server setting
if [[ -z $EXTERNAL_DOWNLOAD_SERVER ]]; then
  echo "It will not use mirror server"
fi

set -e

pushd $ROOT_PATH > /dev/null

export DOCKER_ENV_VARS
export DOCKER_VOLUMES
# Disable nnpackage_run build: mismatch between buildtool for CI and installed hdf5
CMD="export OPTIONS='-DBUILD_NNPACKAGE_RUN=OFF' && \
     export BUILD_TYPE=Release && \
     cp -nv Makefile.template Makefile && \
     make all install build_test_suite"
./nnfw docker-run bash -c "$CMD"

# Model download server setting
if [[ -z $MODELFILE_SERVER ]]; then
  echo "Need model file server setting"
  exit 1
fi

export DOCKER_ENV_VARS=" -e MODELFILE_SERVER=$MODELFILE_SERVER"
./nnfw docker-run-user ./infra/scripts/test_ubuntu_runtime.sh --backend cpu
./nnfw docker-run-user ./infra/scripts/test_ubuntu_runtime.sh --interp

popd > /dev/null
