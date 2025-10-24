#!/bin/bash

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)

if [[ $# -lt 1 ]]; then
  echo "USAGE: $0 ..."
  echo
  echo "ARGUMENTS:"
  echo "  [toolchain.config path]"
  echo "  [Prefix1]"
  echo "  [Prefix2]"
  echo "  ..."
  exit 3
fi

CONFIG_PATH="$1"; shift

source "${CONFIG_PATH}"

echo "-- Use onnx2tflite at '${ONNX2TFLITE_PATH}'"
echo "-- Use nnkit-run at '${NNKIT_RUN_PATH}'"
echo "-- Use ONNX backend: ${ONNX_BACKEND_PATH}"
echo "-- Use TFLITE backend: ${TFLITE_BACKEND_PATH}"
echo "-- Use randomize action: ${RANDOMIZE_ACTION_PATH}"
echo "-- Use HDF5 export action: ${HDF5_EXPORT_ACTION_PATH}"
echo "-- Use HDF5 import action: ${HDF5_IMPORT_ACTION_PATH}"
echo "-- Use i5diff: ${I5DIFF_PATH}"

TESTED=()
PASSED=()
FAILED=()

pushd "${WORKDIR}"

while [[ $# -ne 0 ]]; do
  PREFIX="$1"; shift

  TESTED+=("${PREFIX}")

  PASSED_TAG="${PREFIX}.passed"

  rm -f "${PASSED_TAG}"

  exec 2>&1

  echo "-- Use '${PREFIX}.onnx'"

  # Show commands
  set -x

  # Generate tflite
  "${ONNX2TFLITE_PATH}" \
    "-b" \
    "${WORKDIR}/${PREFIX}.onnx" \
    "${WORKDIR}/${PREFIX}.tflite"

  if [[ $? -ne 0 ]]; then
    continue
  fi

  # Run ONNX
  "${NNKIT_RUN_PATH}" \
    --backend "${ONNX_BACKEND_PATH}" \
    --backend-arg "${WORKDIR}/${PREFIX}.onnx" \
    --pre "${RANDOMIZE_ACTION_PATH}" \
    --pre "${HDF5_EXPORT_ACTION_PATH}" \
    --pre-arg "${WORKDIR}/${PREFIX}.input.h5" \
    --post "${HDF5_EXPORT_ACTION_PATH}" \
    --post-arg "${WORKDIR}/${PREFIX}.expected.h5"

  if [[ $? -ne 0 ]]; then
    continue
  fi

  # Run T/F Lite
  "${NNKIT_RUN_PATH}" \
    --backend "${TFLITE_BACKEND_PATH}" \
    --backend-arg "${WORKDIR}/${PREFIX}.tflite" \
    --pre "${HDF5_IMPORT_ACTION_PATH}" \
    --pre-arg "${WORKDIR}/${PREFIX}.input.h5" \
    --post "${HDF5_EXPORT_ACTION_PATH}" \
    --post-arg "${WORKDIR}/${PREFIX}.obtained.h5"

  if [[ $? -ne 0 ]]; then
    continue
  fi

  "${I5DIFF_PATH}" -d 0.001 "${PREFIX}.expected.h5" "${PREFIX}.obtained.h5"

  if [[ $? -eq 0 ]]; then
    touch "${PASSED_TAG}"
  fi

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
  exit 3
fi

echo "PASSED"
exit 0
