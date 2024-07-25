#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CheckTestPrepared

# TODO Get argument for mix configuration
: ${TEST_ARCH:=$(uname -m | tr '[:upper:]' '[:lower:]')}
TEST_OS="linux"

# nnfw_api_gtest
# NOTE: This test is run here as it does not depend on BACKEND or EXECUTOR

# This test requires test model installation
pushd ${ROOT_PATH} > /dev/null
echo ""
echo "==== Run standalone unittest begin ===="
echo ""
Product/out/test/onert-test unittest --unittestdir=Product/out/unittest --reportdir=report/mixed
echo ""
echo "==== Run standalone unittest end ===="
echo ""

# Test custom op
pushd ${ROOT_PATH} > /dev/null
./Product/out/test/FillFrom_runner
popd > /dev/null
