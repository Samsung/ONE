#!/bin/bash

set -eo pipefail

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd ${CURRENT_PATH}/../../ && pwd)"

# Install path on CI
INSTALL_PATH="$ROOT_PATH/Product/out"

# Install dbus configuration file
DBUS_CONF="${INSTALL_PATH}/share/org.tizen.npud.conf"
mkdir -p /usr/share/dbus-1/system.d/
cp ${DBUS_CONF} /usr/share/dbus-1/system.d/

service dbus restart

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

DEVICE_MODULE_PATH=${INSTALL_PATH}/lib NpudTest
