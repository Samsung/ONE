#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

set -eo pipefail

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

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

source ${CURRENT_PATH}/compiler_modules.sh

NNCC_CFG_OPTION=" -DCMAKE_BUILD_TYPE=Debug"
NNCC_CFG_STRICT=" -DENABLE_STRICT_BUILD=ON"
NNCC_CFG_MODULES=" -DBUILD_WHITELIST=$DEBUG_BUILD_ITEMS"

if [ $# -ne 0 ]; then
	echo "Additional cmake configuration: $@"
fi

./nncc configure $NNCC_CFG_OPTION $NNCC_CFG_STRICT $NNCC_CFG_MODULES
./nncc build -j4

for TEST_BIN in `find ${ROOT_PATH}build/compiler -type f -executable -name *_test`; do
  TEST_NAME="$(basename -- $TEST_BIN)"
  LUGI_LOG=999 $TEST_BIN --gtest_output="xml:$UNITTEST_REPORT_DIR/$TEST_NAME.xml"
done
