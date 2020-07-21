#!/bin/bash

# the following env vars should be defined to call dredd function (except RULE):
#   COMPILED_FILE
#   INSPECT_PROG_PATH
#   VERIFY_PROG_PATH
#   ERROR_LOG

# exit if unknown var is used
set -u

# ---------------
# HELPER FUNCTION

init_error_log()
{
  # create ${ERROR_LOG} that redirect stderr for pipe
  exec 2>"${ERROR_LOG}"
}

argc_check()
{
  ACTUAL_ARGC=$1
  EXPECTED_ARGC=$2

  if [ "$#" -ne 2 ];then
    echo "argc_check : param count must be 2" > ${ERROR_LOG}
    echo "error"  # return value of sub-shell
    exit 1
  fi

  if [ ${ACTUAL_ARGC} -ne ${EXPECTED_ARGC} ];then
    echo "arg count mismatch: actual = ${ACTUAL_ARGC} vs expected = ${EXPECTED_ARGC}" > ${ERROR_LOG}
    echo "error"  # return value of sub-shell
    exit 1
  fi
}

file_path_check()
{
  argc_check $# 1

  if [ ! -f $1 ]; then
    echo "$1 does not exist" > ${ERROR_LOG}
    echo "error"  # return value of sub-shell
    exit 1
  fi
}

check_success_exit_code()
{
  ACTUAL_EXIT_CODE=$1
  EXPECTED_SUCCESS_CODE=$2

  if [ ${ACTUAL_EXIT_CODE} -ne ${EXPECTED_SUCCESS_CODE} ];then
    echo "error"
    exit 1
  fi
}

check_error_exit_code()
{
  ACTUAL_EXIT_CODE=$1
  EXPECTED_ERROR_CODE=$2

  if [ ${ACTUAL_EXIT_CODE} -eq ${EXPECTED_ERROR_CODE} ];then
    echo "error"
    exit 1
  fi
}

# END of HELPER FUNCTION
# ----------------------

#
# Define rule
#
#   - Params: rule name (metric), actual value, condition, expected value
#     - condition is '=', '!=', '<', '>', '<=', '>='. Refer to "man expr"
#   - Return
#     - 0 : success
#     - 1 : fail (condition check fail)
#

RULE()
{
  argc_check $# 4

  RULE_NAME=$1
  ACTUAL=$2
  COND=$3
  EXPECTED=$4

  # not to exit when expr result with 0
  set +e

  expr ${ACTUAL} ${COND} ${EXPECTED} > /dev/null
  RESULT=$?

  # roll-back
  set -e

  # Note: return value of 'expr'
  # - 0 : result is true
  # - 1 : result is false
  # - 2 : error

  if [ ${RESULT} -eq 0 ];then
    echo -e "** [${RULE_NAME}] \t success \t ([actual: ${ACTUAL}] ${COND} [expected: ${EXPECTED}])"
  elif [ ${RESULT} -eq 1 ];then
    echo -e "** [${RULE_NAME}] \t ** fail \t ([actual: ${ACTUAL}] ${COND} [expected: ${EXPECTED}])"
  else
    echo -e "\t** Error in [expr ${ACTUAL} ${COND} ${EXPECTED}]"
  fi

  return ${RESULT}
}

#
# Define each function to get quality value
#

# Note: These function is called by a sub-shell.
# So return value should be passed through "echo return_value"
# tip: for debugging, surround the code with "set -x" and "set +x"

file_size()
{
  file_path_check ${COMPILED_FILE}

  set -o pipefail

  ACTUAL=`init_error_log ; cat ${COMPILED_FILE} | wc -c`

  check_success_exit_code $? 0

  echo ${ACTUAL}
}

all_op_count()
{
  file_path_check ${COMPILED_FILE}
  file_path_check ${INSPECT_PROG_PATH}

  set -o pipefail

  ACTUAL=`init_error_log ; ${INSPECT_PROG_PATH} --operators ${COMPILED_FILE} | wc -l`

  check_success_exit_code $? 0

  echo ${ACTUAL}
}

op_count()
{
  argc_check $# 1
  file_path_check ${COMPILED_FILE}
  file_path_check ${INSPECT_PROG_PATH}

  set -o pipefail

  RESULT=`init_error_log ; ${INSPECT_PROG_PATH} --operators ${COMPILED_FILE}`
  check_success_exit_code $? 0

  # note : grep's exit code is 2 in case of error.
  ACTUAL=`init_error_log ; echo "${RESULT}" | grep -wc "$1"`
  check_error_exit_code $? 2

  echo ${ACTUAL}
}

conv2d_weight_not_constant()
{
  file_path_check ${COMPILED_FILE}
  file_path_check ${INSPECT_PROG_PATH}

  set -o pipefail

  ACTUAL=`init_error_log ; \
          ${INSPECT_PROG_PATH} --conv2d_weight ${COMPILED_FILE} | \
          awk -F, '{ if ($2 != "CONST") print $0}' | wc -l`

  check_success_exit_code $? 0

  echo ${ACTUAL}
}

verify_file_format()
{
  file_path_check ${COMPILED_FILE}
  file_path_check ${VERIFY_PROG_PATH}

  set -o pipefail

  ACTUAL=`init_error_log ; ${VERIFY_PROG_PATH} ${COMPILED_FILE} | grep -c "PASS"`

  # note grep can exit with 1 ("PASS" not found) and this is treated as an error
  check_success_exit_code $? 0

  echo ${ACTUAL}
}

op_version()
{
  argc_check $# 1
  file_path_check ${COMPILED_FILE}
  file_path_check ${INSPECT_PROG_PATH}

  set -o pipefail

  ACTUAL=`init_error_log ; \
          ${INSPECT_PROG_PATH} --op_version ${COMPILED_FILE} | \
          awk -F, -v opname="$1" '{ if ($1 == opname) print $2}'`

  check_success_exit_code $? 0

  echo ${ACTUAL}
}

# TODO define more qullity test function
