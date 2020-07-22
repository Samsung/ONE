#!/bin/bash

# Need at least 2 arguments
if [[ $# -lt 2 ]]; then
  echo "USAGE: $0 ..."
  echo
  echo "ARGUMENTS:"
  echo "  [test.config path]"
  echo "  [WORKDIR]"
  echo "  [Prefix1]"
  echo "  [Prefix2]"
  echo "  ..."
  exit 255
fi

WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$1"; shift
RESOURCE_DIR="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found circle-inspect: ${CIRCLE_INSPECT_PATH}"
echo "-- Found circle-verify: ${CIRCLE_VERIFY_PATH}"
echo "-- Found circle2circle: ${CIRCLE2CIRCLE_PATH}"
echo "-- Found common-artifacts: ${RESOURCE_DIR}"

TESTED=()
PASSED=()
FAILED=()

pushd ${WORKDIR}
while [[ $# -ne 0 ]]; do
  PREFIX="$1"; shift

  TESTED+=("${PREFIX}")

  PASSED_TAG="${PREFIX}.passed"

  rm -f "${PASSED_TAG}"

  cat > "${PREFIX}.log" <(
    exec 2>&1

    echo "-- Found circle: ${PREFIX}.opt.circle"

    # Exit immediately if any command fails
    set -e
    # Show commands
    set -x

    #
    # Check if rule is satisfied
    #

    # Note: turn off 'command printing'. Otherwise printing will be so messy
    set +x

    # (COMPILED_FILE, INSPECT_PROG_PATH, VERIFY_PROG_PATH, ERROR_LOG) must be set for rule-lib.sh
    COMPILED_FILE="${PREFIX}.opt.circle"
    INSPECT_PROG_PATH=${CIRCLE_INSPECT_PATH}
    VERIFY_PROG_PATH=${CIRCLE_VERIFY_PATH}
    ERROR_LOG="${PREFIX}.error"

    rm -f "${ERROR_LOG}"

    # in case error while running rule-lib.sh, prints error msg
    trap 'echo "** ERROR **" ; cat "${ERROR_LOG}"' ERR

    source rule-lib.sh
    source "${RESOURCE_DIR}/${PREFIX}.rule"

    # unset
    trap - ERR
    set -x

    # At this point, the exit code of all commands is 0
    # If not 0, execution of this script ends because of "set -e"
    touch "${PASSED_TAG}"
  )

  if [[ -f "${PASSED_TAG}" ]]; then
    PASSED+=("$PREFIX")
  else
    FAILED+=("$PREFIX")
  fi
done
popd

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
