#!/bin/bash

# This script run commands and check their results
#
# HOW TO USE
#
# ./run_command.sh <path/to/intp_dir>
#                  <TEST NAME #1> <TEST TYPE #1> <CMD TO TEST #1> \
#                  <TEST NAME #2> <TEST TYPE #2> <CMD TO TEST #2> ...
#
# intp_dir    : path to circle-interpreter
# TEST NAME   : identifier for the test which can be printed if it fails
# TEST TYPE   : type of each test cases. it should be one of `POS` or `NEG`
# CMD TO TEST : a command including driver and its arguments 
#                 (ex. sample.circle sample.input sample.output)

INTP_PATH=$1; shift

NUM_TESTS=$#/3
TEST_ARGS=("$@")

TESTED=()
PASSED=()
FAILED=()

for (( i=0; i<${NUM_TESTS}; i++ )); do
  idx=$(( 3*i ))
  TESTNAME="${TEST_ARGS[$idx]}"
  TESTTYPE="${TEST_ARGS["$idx + 1"]}"
  # To replace WORK_DIR env variable to real path
  TESTCMD=$(eval echo ${TEST_ARGS["$idx + 2"]})

  TESTED+=("${TESTNAME}")

  if [ ${TESTTYPE} != "POS" ] && [ ${TESTTYPE} != "NEG" ]; then
    echo "FAILED: ${TESTNAME} type should be \`POS\` or \`NEG\`, but it's ${TESTTYPE}."
    FAILED+=(${TESTNAME})
    continue
  fi

  eval ${INTP_PATH} ${TESTCMD} > /dev/null 2>&1
  
  if [[ $? -eq 0 ]]; then
    if [[ $TESTTYPE == "POS" ]]; then
      PASSED+=("${TESTNAME}")
    else
      FAILED+=("${TESTNAME}")
    fi
  else
    if [[ $TESTTYPE == "NEG" ]]; then
      PASSED+=("${TESTNAME}")
    else
      FAILED+=("${TESTNAME}")
    fi
  fi
done

if [[ ${#TESTED[@]} -ne ${#PASSED[@]} ]]; then
  echo "FAILED"
  for TEST in "${FAILED[@]}"
  do
    echo "- ${TEST}"
  done
  exit 255
fi

echo "PASSED"
exit 0
