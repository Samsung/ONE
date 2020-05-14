#!/bin/bash

if [[ $# -le 2 ]]; then
  echo "USAGE: $0 [mio_tflite_validate path] [prefix 0] "
  exit 255
fi

MIO_TFLITE_VALIDATE_PATH="$1"; shift

echo "-- Found mio_tflite_validate: ${MIO_TFLITE_VALIDATE_PATH}"

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

    echo "'${MIO_TFLITE_VALIDATE_PATH}' '${PREFIX}.tflite'"
    "${MIO_TFLITE_VALIDATE_PATH}" "${PREFIX}.tflite"

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

echo "SUMMARY: ${#PASSED[@]} PASS AND ${#FAILED[@]} FAIL AMONG ${#TESTED[@]} TESTS"

if [[ ${#TESTED[@]} -ne ${#PASSED[@]} ]]; then
  echo "FAILED"
  for TEST in "${FAILED[@]}"
  do
    echo "- ${TEST}"
  done
  exit 255
fi

exit 0
