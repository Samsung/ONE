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

CONFIG_PATH="$1"; shift
WORKDIR="$1"; shift

source "${CONFIG_PATH}"

echo "-- Found TF2CIRCLE: ${TF2CIRCLE_PATH}"
echo "-- Found workdir: ${WORKDIR}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"
while [[ $# -ne 0 ]]; do
  PREFIX="$1"; shift

  TESTED+=("${PREFIX}")

  PASSED_TAG="${PREFIX}.passed"

  rm -f "${PASSED_TAG}"

  cat > "${PREFIX}.log" <(
    exec 2>&1

    echo "-- Found pb: ${PREFIX}.pb"

    # Exit immediately if any command fails
    set -e
    # Show commands
    set -x

    # tflite is generated both for COMPARE and EXPORT actions
    if [ -f "${WORKDIR}/${PREFIX}.customop.conf" ]; then

      # Generate tflite
      "${TF2CIRCLE_PATH}" \
        "${WORKDIR}/${PREFIX}.info" \
        "${WORKDIR}/${PREFIX}.pb" \
        "${WORKDIR}/${PREFIX}.circle" \
        "--customop" "${WORKDIR}/${PREFIX}.customop.conf"
    else

      # Generate circle
      "${TF2CIRCLE_PATH}" \
        "${WORKDIR}/${PREFIX}.info" \
        "${WORKDIR}/${PREFIX}.pb" \
        "${WORKDIR}/${PREFIX}.circle"

    fi

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
