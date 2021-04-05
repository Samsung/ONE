#!/bin/bash

# This script verifies the basic behavior of circle-partitioner
#
# HOW TO USE
#
# ./part_eval_all.sh <path/to/work_dir> <path/to/venv_dir> <path/to/driver> <TEST 1> <TEST 2> ...
#
#    bin_dir  : build directory of circle-part-eval (ex: build/compiler/circle-part-eval)
#    work_dir : artifacts directoy where test materials exist
#    venv_dir : python virtual environment home directory

VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/part_eval_one.py"
WORKDIR="$1"; shift
VIRTUALENV="$1"; shift
CIRCLE_PART_DRIVER_PATH="$1"; shift

TESTED=()
PASSED=()
FAILED=()

for TESTCASE in "$@"; do
  TESTED+=("${TESTCASE}")

  # for simplicity, folder uses same ${TESTCASE}
  TESTCASE_FOLDER="${WORKDIR}/${TESTCASE}"
  
  PASSED_TAG="${TESTCASE_FOLDER}.passed"
  rm -f "${PASSED_TAG}"

  cat > "${TESTCASE_FOLDER}.log" <(
    exec 2>&1
    set -ex

    # chdir into the folder as ini has relative filename of the model
    pushd ${TESTCASE_FOLDER}

    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${VERIFY_SCRIPT_PATH}" \
    --driver "${CIRCLE_PART_DRIVER_PATH}" \
    --name "${TESTCASE}"

    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi

    popd
  )

  if [[ -f "${PASSED_TAG}" ]]; then
    PASSED+=("${TESTCASE}")
  else
    FAILED+=("${TESTCASE}")
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
