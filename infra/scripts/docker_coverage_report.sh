#!/bin/bash

# coverage test data: ${ARCHIVE_PATH}/coverage-data.tar.gz

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

# docker image name
# - for xenial, use DOCKER_IMAGE_NAME="nnfw/one-devtools:xenial"
# - for bionic, use DOCKER_IMAGE_NAME="nnfw/one-devtools:bionic"
if [[ -z $DOCKER_IMAGE_NAME ]]; then
  echo "It will use default docker image name"
fi

NNAS_WORKSPACE=${NNAS_WORKSPACE:-build}
if [[ -z "${ARCHIVE_PATH}" ]]; then
  ARCHIVE_PATH=${NNAS_WORKSPACE}/archive
fi

set -e

pushd $ROOT_PATH > /dev/null

tar -zxf ${ARCHIVE_PATH}/coverage-data.tar.gz

CMD="GCOV_PATH=arm-linux-gnueabihf-gcov NNAS_WORKSPACE=Product ./nnas gen-coverage-report runtime compute &&
     tar -zcf coverage/coverage_report.tar.gz coverage/html &&
     python runtime/3rdparty/lcov-to-cobertura-xml/lcov_cobertura.py coverage/coverage.info -o coverage/nnfw_coverage.xml"

./nnfw docker-run-user bash -c "$CMD"

popd > /dev/null
