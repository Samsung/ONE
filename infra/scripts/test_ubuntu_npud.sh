#!/bin/bash

set -eo pipefail

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd ${CURRENT_PATH}/../../ && pwd)"

# Install path on CI
INSTALL_PATH="$ROOT_PATH/Product/out"
MODEL_PATH="${INSTALL_PATH}/npud-gtest/models"

# Install dbus configuration file
DBUS_CONF="${INSTALL_PATH}/share/org.tizen.npud.conf"
mkdir -p /usr/share/dbus-1/system.d/
cp ${DBUS_CONF} /usr/share/dbus-1/system.d/

service dbus restart

function TestPrepared()
{
  if [[ -z "${MODELFILE}" ]]; then
    echo "Model file is not set. Try to use default setting."
    exit 1
  fi

  mkdir -p ${MODEL_PATH}
  if [[ "${MODELFILE: -7}" == ".tar.gz" ]]; then
    curl -o model.tar.gz -kLsSO ${MODELFILE}
    tar -zxf model.tar.gz -C ${MODEL_PATH}
  else
    echo "The file format is not supported."
    echo "Supported format: tar.gz"
    exit 1
  fi
}

function TestCleanUp()
{
  rm -rf ${MODEL_PATH}
}

function NpudTest()
{
  pushd ${ROOT_PATH} > /dev/null

  $INSTALL_PATH/npud-gtest/npud_gtest
  EXITCODE=$?
  if [ ${EXITCODE} -ne 0 ]; then
    exit ${EXITCODE}
  fi

  popd > /dev/null
}

TestPrepared

DEVICE_MODULE_PATH=${INSTALL_PATH}/lib GTEST_MODEL_PATH=${MODEL_PATH} NpudTest

TestCleanUp
