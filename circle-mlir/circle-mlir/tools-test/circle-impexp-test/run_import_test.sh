#!/bin/bash

# This script is to check import module by running import and then export
# with circle-impexp tool, that should run without any problem.
#
# HOW TO USE
#
# ./run_import_test.sh <model>
# model    : circle model base name

set -e

TEST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="$1"; shift

CIRCLE_SOURCE_FILE="${MODEL_NAME}.circle"
CIRCLE_TARGET_FILE="${MODEL_NAME}.2.circle"

COMP_RESULT=0

if [[ -f ${CIRCLE_TARGET_FILE} && -s ${CIRCLE_TARGET_FILE} ]]; then
  # NOTE for initial version just check file exist and not zero length
  # TODO do more validation
  echo "${CIRCLE_TARGET_FILE} looks OK"
else
  COMP_RESULT=1
fi

exit ${COMP_RESULT}
