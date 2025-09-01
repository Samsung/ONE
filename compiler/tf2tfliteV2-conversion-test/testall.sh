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

echo "-- Found TF2TFLITEV2: ${TF2TFLITEV2_PATH}"
echo "-- Found python virtualenv: ${VIRTUALENV}"

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

    echo "-- Found pbtxt: ${PREFIX}.pbtxt"

    # Exit immediately if any command fails
    set -e
    # Show commands
    set -x

    # Generate tflite
    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${TF2TFLITEV2_PATH}" \
      --input_path "${WORKDIR}/${PREFIX}.pbtxt" \
      --input_arrays "$(awk -F, '/^input/ { print $2 }' ${PREFIX}.info | cut -d: -f1 | tr -d ' ' | paste -d, -s)" \
      --input_shapes "$(cat ${PREFIX}.info | grep '^input' | cut -d '[' -f2 | cut -d ']' -f1 | tr -d ' ' | xargs | tr ' ' ':')" \
      --output_path "${WORKDIR}/${PREFIX}.tflite" \
      --output_arrays "$(awk -F, '/^output/ { print $2 }' ${PREFIX}.info | cut -d: -f1 | tr -d ' ' | paste -d, -s)"

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
