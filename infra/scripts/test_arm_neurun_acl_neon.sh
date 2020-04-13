#!/bin/bash

###
### This script will be deprecated
###

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

set -eo pipefail

# Test tflite_loader
pushd ${ROOT_PATH} > /dev/null

./infra/scripts/test_ubuntu_runtime.sh --backend acl_neon

# TODO Independent test when supporting ops are increased
export NCNN_LAYOUT="NCHW"
./infra/scripts/test_ubuntu_runtime.sh --backend srcn --linear-only
unset NCNN_LAYOUT

popd > /dev/null
