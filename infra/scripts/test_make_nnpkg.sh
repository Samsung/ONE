#!/bin/bash

# Resource: test-resources.tar.gz
# nncc package: nncc-package.tar.gz

# Test suite: nnpkg-test-suite.tar.gz

set -eo pipefail

NNAS_WORKSPACE=${NNAS_WORKSPACE:-build}
if [[ -z "${ARCHIVE_PATH}" ]]; then
  ARCHIVE_PATH=${NNAS_WORKSPACE}/archive
  echo "Default archive directory including nncc package and resources: ${ARCHIVE_PATH}"
fi

pushd ${ROOT_PATH} > /dev/null

RESOURCE_PATH=${NNAS_WORKSPACE}/tfmodel
BIN_PATH=${NNAS_WORKSPACE}/bin/nncc
mkdir -p ${BIN_PATH}
mkdir -p ${RESOURCE_PATH}
tar -zxf ${ARCHIVE_PATH}/nncc-package.tar.gz -C ${BIN_PATH}
tar -zxf ${ARCHIVE_PATH}/test-resources.tar.gz -C ${RESOURCE_PATH}

export PATH=${PATH}:${PWD}/${BIN_PATH}/bin

for f in `find ${RESOURCE_PATH} -name "*.pb" | cut -d'.' -f1 | sort | uniq`;
do
  tools/nnpackage_tool/nncc-tc-to-nnpkg-tc/nncc-tc-to-nnpkg-tc.sh -o nnpkg-tcs -i ${f%/*} $(basename $f);
done

tar -zcf ${ARCHIVE_PATH}/nnpkg-test-suite.tar.gz nnpkg-tcs

popd > /dev/null
