#!/bin/bash

set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CheckTestPrepared

pushd ${ROOT_PATH}

# NOTE Fixed backend assignment by type of operation
# TODO Enhance this with randomized test
BACKENDS=(acl_cl acl_neon cpu)

# Get the intersect of framework test list files
TESTLIST_PREFIX="tests/scripts/list/neurun_frameworktest_list.aarch64"
SKIPLIST_PREFIX="Product/out/unittest/nnapi_gtest.skip.aarch64-linux"
sort $TESTLIST_PREFIX.${BACKENDS[0]}.txt > $TESTLIST_PREFIX.intersect.txt
sort $SKIPLIST_PREFIX.${BACKENDS[0]} > $SKIPLIST_PREFIX.union
for BACKEND in "${BACKENDS[@]:1}"; do
    comm -12 <(sort $TESTLIST_PREFIX.intersect.txt) <(sort $TESTLIST_PREFIX.$BACKEND.txt) > $TESTLIST_PREFIX.intersect.next.txt
    comm <(sort $SKIPLIST_PREFIX.union) <(sort $SKIPLIST_PREFIX.$BACKEND) | tr -d "[:blank:]" > $SKIPLIST_PREFIX.union.next
    mv $TESTLIST_PREFIX.intersect.next.txt $TESTLIST_PREFIX.intersect.txt
    mv $SKIPLIST_PREFIX.union.next $SKIPLIST_PREFIX.union
done
popd > /dev/null

# Run the test
export OP_BACKEND_Conv2D="cpu"
export OP_BACKEND_MaxPool2D="acl_cl"
export OP_BACKEND_AvgPool2D="acl_neon"
export ACL_LAYOUT="NCHW"
export NCNN_LAYOUT="NCHW"
Unittests "acl_cl;acl_neon;cpu" "Product/out/unittest/nnapi_gtest.skip.aarch64-linux.union" "report/mixed"
TFLiteModelVerification "acl_cl;acl_neon;cpu" "${TESTLIST_PREFIX}.intersect.txt" "report/mixed"
