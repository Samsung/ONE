#!/usr/bin/env bash

set -eo pipefail

FLATC_BIN=${FLATC_BIN:-}
# FLATC_BIN
if [ -z "${FLATC_BIN}" ]; then
  pushd ../../
  make -f Makefile.template prepare-buildtool
  popd || exit
  FLATC_BIN='../../Product/buildtool/out/bin/flatc'
fi

${FLATC_BIN} --python -o src ../../res/TensorFlowLiteSchema/2.19.0/schema.fbs
