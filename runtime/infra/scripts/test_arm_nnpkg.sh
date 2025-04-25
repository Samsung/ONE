#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BACKENDS=("acl_cl" "acl_neon" "cpu")

for BACKEND in "${BACKENDS[@]}";
do
  NNPackageTest ${BACKEND} "Product/out/test/list/nnpkg_test_list.armv7l-linux.${BACKEND}"
done

unset DISABLE_COMPILE
