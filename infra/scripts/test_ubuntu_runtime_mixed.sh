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
echo ""
echo "==== Run standalone unittest begin ===="
echo ""
Product/out/test/onert-test prepare-model --model=nnpackage
Product/out/test/onert-test unittest --unittestdir=Product/out/unittest_standalone
echo ""
echo "==== Run standalone unittest end ===="
echo ""

# Test custom op
pushd ${ROOT_PATH} > /dev/null
./Product/out/test/FillFrom_runner
popd > /dev/null

# NOTE Fixed backend assignment by type of operation
# TODO Enhance this with randomized test
BACKENDS=(acl_cl acl_neon cpu)

# Get the intersect of framework test list files
TESTLIST_PREFIX="Product/out/test/list/frameworktest_list.${TEST_ARCH}"
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
NNAPIGTest "acl_cl;acl_neon;cpu" "Product/out/unittest/nnapi_gtest.skip.${TEST_ARCH}-${TEST_OS}.union" "report/mixed"
TFLiteModelVerification "acl_cl;acl_neon;cpu" "${TESTLIST_PREFIX}.intersect.txt" "report/mixed"
