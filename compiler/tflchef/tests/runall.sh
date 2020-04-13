#!/bin/bash

if [[ $# -le 3 ]]; then
  echo "USAGE: $0 [nnkit-run path] [tflite backend path] [working directory] [prefix 0] [prefix 1] ..."
  exit 255
fi

NNKIT_RUN_PATH="$1"; shift
TFLITE_BACKEND_PATH="$1"; shift
WORKDIR="$1"; shift

echo "-- Found nnkit-run: ${NNKIT_RUN_PATH}"
echo "-- Found tflite backend: ${TFLITE_BACKEND_PATH}"
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

    echo "'${NNKIT_RUN_PATH}' --backend '${TFLITE_BACKEND_PATH}' --backend-arg '${PREFIX}.tflite'"
    "${NNKIT_RUN_PATH}" --backend "${TFLITE_BACKEND_PATH}" --backend-arg "${PREFIX}.tflite"

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
