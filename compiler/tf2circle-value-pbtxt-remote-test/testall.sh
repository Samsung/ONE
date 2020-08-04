#!/bin/bash

# Need at least 4 arguments
if [[ $# -lt 4 ]]; then
  echo "USAGE: $0 ..."
  echo
  echo "ARGUMENTS:"
  echo "  [test.config path]"
  echo "  [WORKDIR]"
  echo "  [REMOTE_IP]"
  echo "  [REMOTE_USER]"
  echo "  [Prefix1]"
  echo "  [Prefix2]"
  echo "  ..."
  exit 255
fi

CONFIG_PATH="$1"; shift
WORKDIR="$1"; shift
REMOTE_IP="$1"; shift
REMOTE_USER="$1"; shift

CURRENT_DATETIME=$(date +'%Y%m%d_%H%M%S')
REMOTE_WORKDIR=${REMOTE_WORKDIR:-"CVT_${CURRENT_DATETIME}"}
RESULT_CSV="${WORKDIR}/Result_${CURRENT_DATETIME}.csv"

source "${CONFIG_PATH}"

echo "-- Found nnkit-run: ${NNKIT_RUN_PATH}"
echo "-- Found TF backend: ${TF_BACKEND_PATH}"
echo "-- Found TF2CIRCLE: ${TF2CIRCLE_PATH}"
echo "-- Found MODEL2NNPKG: ${MODEL2NNPKG_PATH}"
echo "-- Found Runtime library: ${RUNTIME_LIBRARY_PATH}"
echo "-- Found randomize action: ${RANDOMIZE_ACTION_PATH}"
echo "-- Found HDF5 export action: ${HDF5_EXPORT_ACTION_PATH}"
echo "-- Found HDF5 import action: ${HDF5_IMPORT_ACTION_PATH}"
echo "-- Found workdir: ${WORKDIR}"

if [ -z ${MODEL2NNPKG_PATH} ] || [ ! -f ${MODEL2NNPKG_PATH} ]; then
  echo "MODEL2NNPKG is not found"
  exit 3
fi

# Register remote machine ssh information
cat /dev/zero | ssh-keygen -q -N ""
ssh-copy-id -o ConnectTimeout=5 "${REMOTE_USER}@${REMOTE_IP}"

# Odroid IP address validation
if [[ $? -ne 0 ]]; then
  echo "Cannot reach to given remote machine. Check IP address or username."
  exit 5
fi

# Send runtime library files
ssh "${REMOTE_USER}@${REMOTE_IP}" "mkdir -p ${REMOTE_WORKDIR}/Product/"
scp -r "${RUNTIME_LIBRARY_PATH}" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_WORKDIR}/Product/"

TESTED=()
PASSED=()
FAILED=()
echo "TEST_NAME, TF2CIRCLE, CIRCLE_VALUE_TEST" >> ${RESULT_CSV}

pushd "${WORKDIR}"
while [[ $# -ne 0 ]]; do
  PREFIX="$1"; shift

  TESTED+=("${PREFIX}")

  PASSED_TAG="${PREFIX}.passed"

  rm -f "${PASSED_TAG}" "${PREFIX}.circle"

  # Information to be recorded
  TF2CIRCLE_PASSED=FALSE
  CIRCLE_VALUE_PASSED=FALSE

  cat > "${PREFIX}.log" <(
    exec 2>&1

    echo "-- Found pb: ${PREFIX}.pb"

    # Exit immediately if any command fails
    set -e
    # Show commands
    set -x

    # Generate circle
    "${TF2CIRCLE_PATH}" \
      "${WORKDIR}/${PREFIX}.info" \
      "${WORKDIR}/${PREFIX}.pb" \
      "${WORKDIR}/${PREFIX}.circle"

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

    # Generate nnpackage model
    "${MODEL2NNPKG_PATH}" -o "${WORKDIR}" "${WORKDIR}/${PREFIX}.circle"

    # Copy h5 files into nnpackage
    mkdir -p "${WORKDIR}/${PREFIX}/metadata/tc"
    cp "${WORKDIR}/${PREFIX}.input.h5" "${WORKDIR}/${PREFIX}/metadata/tc/input.h5"
    cp "${WORKDIR}/${PREFIX}.expected.h5" "${WORKDIR}/${PREFIX}/metadata/tc/expected.h5"

    # Run test_arm_nnpkg in remote machine
    scp -r "${WORKDIR}/${PREFIX}/" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_WORKDIR}/${PREFIX}/"
    ssh "${REMOTE_USER}@${REMOTE_IP}" "cd ${REMOTE_WORKDIR}; ./Product/out/test/onert-test nnpkg-test -i . -o  ${PREFIX}/metadata/tc ${PREFIX}"

    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
  )

  if [[ -f "${PREFIX}.circle" ]]; then
    TF2CIRCLE_PASSED=TRUE
  else
    TF2CIRCLE_PASSED=FALSE
  fi

  if [[ -f "${PASSED_TAG}" ]]; then
    PASSED+=("$PREFIX")
    CIRCLE_VALUE_PASSED=TRUE
  else
    FAILED+=("$PREFIX")
    CIRCLE_VALUE_PASSED=FALSE
  fi

  echo "${PREFIX}, ${TF2CIRCLE_PASSED}, ${CIRCLE_VALUE_PASSED}" >> ${RESULT_CSV}
done
popd

rm -f Result_latest
ln -s ${RESULT_CSV} Result_latest

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
