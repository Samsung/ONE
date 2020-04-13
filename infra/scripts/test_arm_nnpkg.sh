#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKENDS=("acl_cl" "acl_neon" "cpu")

for BACKEND in "${BACKENDS[@]}";
do
  NNPackageTest ${BACKEND} "tools/nnpackage_tool/nnpkg_test/list.armv7l-linux.${BACKEND}"
done

# Interpreter test
export DISABLE_COMPILE=1
NNPackageTest "interp" "tools/nnpackage_tool/nnpkg_test/list.noarch.interp"
unset DISABLE_COMPILE
