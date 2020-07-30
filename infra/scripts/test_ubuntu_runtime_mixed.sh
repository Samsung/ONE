#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CheckTestPrepared

# TODO Get argument for mix configuration
TEST_ARCH=$(uname -m | tr '[:upper:]' '[:lower:]')
TEST_OS="linux"

# nnfw_api_gtest
# NOTE: This test is run here as it does not depend on BACKEND or EXECUTOR

# This test requires test model installation
pushd ${ROOT_PATH} > /dev/null
echo
echo "==== Install nnpackage for nnfw_api_gtest begin ===="
echo

NNFW_API_TEST_MODEL_INSTALLER=tests/scripts/nnfw_api_gtest/install_nnfw_api_gtest_nnpackages.sh
TEST_BIN=Product/out/unittest_standalone/nnfw_api_gtest
$NNFW_API_TEST_MODEL_INSTALLER --install-dir ${TEST_BIN}_models

echo
echo "==== Install nnpackage for nnfw_api_gtest end ===="
echo

Product/out/test/onert-test unittest --reportdir=report --unittestdir=Product/out/unittest_standalone
popd > /dev/null

pushd ${ROOT_PATH}

# NOTE Fixed backend assignment by type of operation
# TODO Enhance this with randomized test
BACKENDS=(acl_cl acl_neon cpu)

# Get the intersect of framework test list files
TESTLIST_PREFIX="tests/scripts/list/frameworktest_list.${TEST_ARCH}"
SKIPLIST_PREFIX="Product/out/unittest/nnapi_gtest.skip.${TEST_ARCH}-${TEST_OS}"
sort $TESTLIST_PREFIX.${BACKENDS[0]}.txt > $TESTLIST_PREFIX.intersect.txt
sort $SKIPLIST_PREFIX.${BACKENDS[0]} > $SKIPLIST_PREFIX.union
for BACKEND in "${BACKENDS[@]:1}"; do
    comm -12 <(sort $TESTLIST_PREFIX.intersect.txt) <(sort $TESTLIST_PREFIX.$BACKEND.txt) > $TESTLIST_PREFIX.intersect.next.txt
    comm <(sort $SKIPLIST_PREFIX.union) <(sort $SKIPLIST_PREFIX.$BACKEND) | tr -d "[:blank:]" > $SKIPLIST_PREFIX.union.next
    mv $TESTLIST_PREFIX.intersect.next.txt $TESTLIST_PREFIX.intersect.txt
    mv $SKIPLIST_PREFIX.union.next $SKIPLIST_PREFIX.union
done
popd > /dev/null

# Fail on NCHW layout (acl_cl, acl_neon)
# TODO Fix bug
echo "GeneratedTests.*weights_as_inputs*" >> $SKIPLIST_PREFIX.union
echo "GeneratedTests.logical_or_broadcast_4D_2D_nnfw" >> $SKIPLIST_PREFIX.union
echo "GeneratedTests.mean" >> $SKIPLIST_PREFIX.union
echo "GeneratedTests.add_broadcast_4D_2D_after_nops_float_nnfw" >> $SKIPLIST_PREFIX.union
echo "GeneratedTests.argmax_*" >> $SKIPLIST_PREFIX.union
echo "GeneratedTests.squeeze_relaxed" >> $SKIPLIST_PREFIX.union

# Run the test
export OP_BACKEND_Conv2D="cpu"
export OP_BACKEND_MaxPool2D="acl_cl"
export OP_BACKEND_AvgPool2D="acl_neon"
export ACL_LAYOUT="NCHW"
Unittests "acl_cl;acl_neon;cpu" "Product/out/unittest/nnapi_gtest.skip.${TEST_ARCH}-${TEST_OS}.union" "report/mixed"
TFLiteModelVerification "acl_cl;acl_neon;cpu" "${TESTLIST_PREFIX}.intersect.txt" "report/mixed"
