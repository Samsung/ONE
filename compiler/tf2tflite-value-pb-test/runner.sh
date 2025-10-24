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

echo "-- Found nnkit-run: ${NNKIT_RUN_PATH}"
echo "-- Found TF backend: ${TF_BACKEND_PATH}"
echo "-- Found TFLITE backend: ${TFLITE_BACKEND_PATH}"
echo "-- Found TF2TFLITE: ${TF2TFLITE_PATH}"
echo "-- Found randomize action: ${RANDOMIZE_ACTION_PATH}"
echo "-- Found HDF5 export action: ${HDF5_EXPORT_ACTION_PATH}"
echo "-- Found HDF5 import action: ${HDF5_IMPORT_ACTION_PATH}"
echo "-- Found i5diff: ${I5DIFF_PATH}"
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

    # Generate tflite
    "${TF2TFLITE_PATH}" \
      "${MODEL_INFO_PATH}" \
      "${MODEL_PB_PATH}" \
      "${WORKDIR}/${PREFIX}.tflite"

    # Run TensorFlow
    "${NNKIT_RUN_PATH}" \
      --backend "${TF_BACKEND_PATH}" \
      --backend-arg "${MODEL_PB_PATH}" \
      --backend-arg "${MODEL_INFO_PATH}" \
      --pre "${RANDOMIZE_ACTION_PATH}" \
      --pre "${HDF5_EXPORT_ACTION_PATH}" \
      --pre-arg "${WORKDIR}/${PREFIX}.input.h5" \
      --post "${HDF5_EXPORT_ACTION_PATH}" \
      --post-arg "${WORKDIR}/${PREFIX}.expected.h5"

    # Run TensorFlow Lite
    "${NNKIT_RUN_PATH}" \
      --backend "${TFLITE_BACKEND_PATH}" \
      --backend-arg "${WORKDIR}/${PREFIX}.tflite" \
      --pre "${HDF5_IMPORT_ACTION_PATH}" \
      --pre-arg "${WORKDIR}/${PREFIX}.input.h5" \
      --post "${HDF5_EXPORT_ACTION_PATH}" \
      --post-arg "${WORKDIR}/${PREFIX}.obtained.h5"

    "${I5DIFF_PATH}" -d 0.001 "${PREFIX}.expected.h5" "${PREFIX}.obtained.h5"

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
