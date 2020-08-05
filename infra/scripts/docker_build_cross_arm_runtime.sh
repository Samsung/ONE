#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

# prepare rootfs
if [ -z "$ROOTFS_DIR" ] || [ ! -d $ROOTFS_DIR ]; then
  echo "It will use default rootfs path"
else
  DOCKER_VOLUMES+=" -v $ROOTFS_DIR:/opt/rootfs"
  DOCKER_ENV_VARS+=" -e ROOTFS_DIR=/opt/rootfs"
fi

# mount volume (or directory) for externals
if [ -n "$EXTERNAL_VOLUME" ]; then
  DOCKER_VOLUMES+=" -v $EXTERNAL_VOLUME:/externals"
  DOCKER_ENV_VARS+=" -e EXTERNAL_VOLUME=/externals"
else
  echo "It will use default external path"
fi

# docker image name
if [[ -z $DOCKER_IMAGE_NAME ]]; then
  echo "It will use default docker image name"
fi

# Mirror server setting
if [[ -z $EXTERNAL_DOWNLOAD_SERVER ]]; then
  echo "It will not use mirror server"
fi

DOCKER_ENV_VARS+=" -e TARGET_ARCH=armv7l"
DOCKER_ENV_VARS+=" -e CROSS_BUILD=1"

set -e

pushd $ROOT_PATH > /dev/null

# TODO use command instead of makefile
export DOCKER_ENV_VARS
export DOCKER_VOLUMES
CMD="cp -nv Makefile.template Makefile && \
     make all install build_test_suite"
./nnfw docker-run bash -c "$CMD"

popd > /dev/null
