#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKENDS=("acl_cl" "acl_neon" "cpu")

for BACKEND in "${BACKENDS[@]}";
do
  NNPackageTest ${BACKEND} "tests/scripts/list/nnpkg_test_list.armv7l-linux.${BACKEND}"
done

# Interpreter test
export DISABLE_COMPILE=1
NNPackageTest "interp" "tests/scripts/list/nnpkg_test_list.noarch.interp"
unset DISABLE_COMPILE
