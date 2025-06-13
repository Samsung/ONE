#!/bin/bash

# TODO add check arguments

VIRTUAL_ENV=$1; shift
TEST_LIST_FILE=$1; shift
ARTIFACTS_BIN_PATH=$1; shift
CMAKE_CURRENT_BINARY_DIR=$1; shift
LUCI_EVAL_DRIVER=$1; shift

source ${VIRTUAL_ENV}/bin/activate

python -m pytest -sv test_luci_eval.py \
  --test_list ${TEST_LIST_FILE} \
  --tflite_dir ${ARTIFACTS_BIN_PATH} \
  --circle_dir ${CMAKE_CURRENT_BINARY_DIR} \
  --luci_eval_driver ${LUCI_EVAL_DRIVER}

deactivate
