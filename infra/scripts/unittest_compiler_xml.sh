#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

set -eo pipefail

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"
NNCC_WORKSPACE=${NNCC_WORKSPACE:-${ROOT_PATH}build}

# Use fixed absolute report dir for CI
UNITTEST_REPORT_DIR=${ROOT_PATH}build/unittest_compiler_xml

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
  TEST_DIR="$(dirname $TEST_BIN)"

  # Execute on test directory to find related file
  pushd $TEST_DIR > /dev/null
  LUGI_LOG=999 ./$TEST_NAME --gtest_output="xml:$UNITTEST_REPORT_DIR/$TEST_NAME.xml"
  popd > /dev/null
done
