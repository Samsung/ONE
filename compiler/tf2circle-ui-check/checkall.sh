#!/bin/bash

# USAGE: check_all.sh [CONFIG] [TEST 1] [TEST 2] ...
CONFIG_PATH="$1"; shift

source "${CONFIG_PATH}"

echo "###"
echo "### tf2circle UI check"
echo "###"
echo

echo "Get each test from '${TESTCASE_BASE}/'"
echo "Use tfkit at '${TFKIT_PATH}'"
echo "Use tf2circle at '${TF2CIRCLE_PATH}'"
echo

while [[ $# -ne 0 ]]; do
  NAME="$1"; shift
  TESTCASE_DIR="${TESTCASE_BASE}/${NAME}"

  INFO_FILE="${TESTCASE_DIR}/test.info"
  PBTXT_FILE="${TESTCASE_DIR}/test.pbtxt"
  MANIFEST_FILE="${TESTCASE_DIR}/test.manifest"

  PB_FILE="${NAME}.pb"


  echo "Running '${NAME}'"
  if [[ -f ${MANIFEST_FILE} ]]; then
    # TODO Only dump SUMMARY
    cat ${MANIFEST_FILE}
  fi
  echo

  # Create a pb model
  "${TFKIT_PATH}" encode "${PBTXT_FILE}" "${PB_FILE}"

  echo "OUTPUT:"
  echo "---------------------------------------------------------"
  # Generate circle
  "${TF2CIRCLE_PATH}" "${INFO_FILE}" "${PB_FILE}" "${NAME}.circle"
  EXITCODE=$?
  echo "---------------------------------------------------------"

  echo
  echo "EXITCODE: ${EXITCODE}"

  echo "Running '${NAME}' - Done"
done

echo
echo "###"
echo "### tf2circle UI check (done)"
echo "###"

exit 0
