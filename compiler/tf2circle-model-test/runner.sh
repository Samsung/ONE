#!/bin/bash

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)

# Need at least toolchain.config
if [[ $# -lt 1 ]]; then
  echo "USAGE: $0 ..."
  echo
  echo "ARGUMENTS:"
  echo "  [toolchain.config path]"
  echo "  [Prefix1]"
  echo "  [Prefix2]"
  echo "  ..."
  exit 255
fi

CONFIG_PATH="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found TF2CIRCLE: ${TF2CIRCLE_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"
while [[ $# -ne 0 ]]; do
  PREFIX="$1"; shift

  echo "[ RUN      ] ${PREFIX}"

  TESTED+=("${PREFIX}")

  PASSED_TAG="${PREFIX}.passed"

  rm -f "${PASSED_TAG}"

  cat > "${PREFIX}.log" <(
    exec 2>&1

    source "${PREFIX}.test"

    echo "-- Use '${MODEL_PB_PATH}' and '${MODEL_INFO_PATH}'"

    # Exit immediately if any command fails
    set -e
    # Show commands
    set -x

    # Generate circle
    "${TF2CIRCLE_PATH}" \
      "${MODEL_INFO_PATH}" \
      "${MODEL_PB_PATH}" \
      "${WORKDIR}/${PREFIX}.circle"

    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
  )

  if [[ -f "${PASSED_TAG}" ]]; then
    echo "[       OK ] ${PREFIX}"
    PASSED+=("$PREFIX")
  else
    echo "[      FAIL] ${PREFIX}"
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
