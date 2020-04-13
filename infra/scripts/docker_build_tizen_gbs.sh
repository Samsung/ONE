#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

GBS_RPM_DIR=$ROOT_PATH/Product/out/rpm
mkdir -p $GBS_RPM_DIR
DOCKER_VOLUMES=" -v $GBS_RPM_DIR:/opt/rpm"

if [[ -z $DOCKER_IMAGE_NAME ]]; then
  echo "It will use default docker image name for tizen gbs build"
  DOCKER_IMAGE_NAME="nnfw_docker_tizen"
fi

DOCKER_ENV_VARS=" --privileged"

set -e

pushd $ROOT_PATH > /dev/null

CMD="gbs -c $ROOT_PATH/infra/nnfw/config/gbs.conf build \
         -A armv7l --profile=profile.tizen --clean --include-all --define '$GBS_DEFINE' && \
     cp -rf /home/GBS-ROOT/local/repos/tizen/armv7l/RPMS/*.rpm /opt/rpm/"

export DOCKER_ENV_VARS
export DOCKER_VOLUMES
./nnfw docker-run bash -c "$CMD"

popd > /dev/null
