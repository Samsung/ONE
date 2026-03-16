#!/bin/bash

if [[ $# -le 6 ]]; then
  echo "USAGE: $0 [nnkit-run path] [reference backend path] [randomize action path] [HDF5 export action path] [HDF5 import action path] [WORKDIR] [Prefix1] [Prefix2] ..."
  exit 255
fi

NNKIT_RUN_PATH="$1"; shift
REFERENCE_BACKEND_PATH="$1"; shift
RANDOMIZE_ACTION_PATH="$1"; shift
HDF5_EXPORT_ACTION_PATH="$1"; shift
HDF5_IMPORT_ACTION_PATH="$1"; shift
WORKDIR="$1"; shift

echo "-- Found nnkit-run: ${NNKIT_RUN_PATH}"
echo "-- Found reference backend: ${REFERENCE_BACKEND_PATH}"
echo "-- Found randomize action: ${RANDOMIZE_ACTION_PATH}"
echo "-- Found HDF5 export action: ${HDF5_EXPORT_ACTION_PATH}"
echo "-- Found HDF5 import action: ${HDF5_IMPORT_ACTION_PATH}"
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

    echo "-- Found tflite: ${PREFIX}.tflite"
    echo "-- Found backend: lib${PREFIX}.so"

    "${NNKIT_RUN_PATH}" \
      --backend "${REFERENCE_BACKEND_PATH}" \
      --backend-arg "${WORKDIR}/${PREFIX}.tflite" \
      --pre "${RANDOMIZE_ACTION_PATH}" \
      --pre "${HDF5_EXPORT_ACTION_PATH}" \
      --pre-arg "${PREFIX}.input.h5" \
      --post "${HDF5_EXPORT_ACTION_PATH}" \
      --post-arg "${PREFIX}.expected.h5"

    "${NNKIT_RUN_PATH}" \
      --backend "./lib${PREFIX}.so" \
      --pre "${HDF5_IMPORT_ACTION_PATH}" \
      --pre-arg "${PREFIX}.input.h5" \
      --post "${HDF5_EXPORT_ACTION_PATH}" \
      --post-arg "${PREFIX}.obtained.h5"

    h5diff -d 0.001 "${PREFIX}.expected.h5" "${PREFIX}.obtained.h5"

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
