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

BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$1"; shift
WORKDIR="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found TFLITE2CIRCLE: ${TFLITE2CIRCLE_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"
while [[ $# -ne 0 ]]; do
  PREFIX="$1"; shift

  TESTED+=("${PREFIX}")

  PASSED_TAG="${BINDIR}/${PREFIX}.passed"

  rm -f "${PASSED_TAG}"

  cat > "${BINDIR}/${PREFIX}.log" <(
    exec 2>&1

    echo "-- Found tflite: ${PREFIX}.tflite"

    # Exit immediately if any command fails
    set -e
    # Show commands
    set -x

    # Generate circle
    "${TFLITE2CIRCLE_PATH}" \
      "${WORKDIR}/${PREFIX}.tflite" \
      "${BINDIR}/${PREFIX}.circle"

    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
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
