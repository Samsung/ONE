#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

set -eo pipefail

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"
NNCC_WORKSPACE=${NNCC_WORKSPACE:-${ROOT_PATH}build}
UNITTEST_REPORT_DIR=${NNCC_WORKSPACE}/unittest_compiler_xml

for i in "$@"
do
  case $i in
    --reportdir=*)
      UNITTEST_REPORT_DIR=${i#*=}
      ;;
  esac
  shift
done

if [ ! -e "$UNITTEST_REPORT_DIR" ]; then
  mkdir -p $UNITTEST_REPORT_DIR
fi

for TEST_BIN in `find ${NNCC_WORKSPACE}/compiler -type f -executable -name *_test`; do
  TEST_NAME="$(basename -- $TEST_BIN)"
  LUGI_LOG=999 $TEST_BIN --gtest_output="xml:$UNITTEST_REPORT_DIR/$TEST_NAME.xml"
done
