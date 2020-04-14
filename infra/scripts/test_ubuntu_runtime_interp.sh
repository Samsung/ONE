#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export DISABLE_COMPILE=1
CheckTestPrepared
echo "[[ Interpreter test ]]"
Unittests "cpu" "Product/out/unittest/nnapi_gtest.skip.noarch.interp" "report/interp"
TFLiteModelVerification "cpu" "tests/scripts/list/frameworktest_list.noarch.interp.txt" "report/interp"

unset DISABLE_COMPILE
