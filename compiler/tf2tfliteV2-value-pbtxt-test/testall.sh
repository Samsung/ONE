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

echo "-- Found nnkit-run: ${NNKIT_RUN_PATH}"
echo "-- Found TF backend: ${TF_BACKEND_PATH}"
echo "-- Found TFLITE backend: ${TFLITE_BACKEND_PATH}"
echo "-- Found TF2TFLITEV2: ${TF2TFLITEV2_PATH}"
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

    # Generate tflite
    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${TF2TFLITEV2_PATH}" \
      --v1 \
      --input_path "${WORKDIR}/${PREFIX}.pb" \
      --input_arrays "$(awk -F, '/^input/ { print $2 }' ${PREFIX}.info | cut -d: -f1 | tr -d ' ' | paste -d, -s)" \
      --input_shapes "$(cat ${PREFIX}.info | grep '^input' | cut -d '[' -f2 | cut -d ']' -f1 | tr -d ' ' | xargs | tr ' ' ':')" \
      --output_path "${WORKDIR}/${PREFIX}.tflite" \
      --output_arrays "$(awk -F, '/^output/ { print $2 }' ${PREFIX}.info | cut -d: -f1 | tr -d ' ' | paste -d, -s)"

    # Run TensorFlow
    "${NNKIT_RUN_PATH}" \
      --backend "${TF_BACKEND_PATH}" \
      --backend-arg "${WORKDIR}/${PREFIX}.pb" \
      --backend-arg "${WORKDIR}/${PREFIX}.info" \
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
