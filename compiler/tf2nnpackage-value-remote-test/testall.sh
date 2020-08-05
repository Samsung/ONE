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

BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CURRENT_DATETIME=$(date +'%Y%m%d_%H%M%S')
REMOTE_WORKDIR=${REMOTE_WORKDIR:-"CVT_${CURRENT_DATETIME}"}
RESULT_CSV="${BINDIR}/Result_${CURRENT_DATETIME}.csv"

source "${CONFIG_PATH}"

echo "-- Found Runtime library: ${RUNTIME_LIBRARY_PATH}"
echo "-- Found workdir: ${WORKDIR}"

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
echo "TEST_NAME, CIRCLE_VALUE_TEST" >> ${RESULT_CSV}

pushd "${WORKDIR}"
while [[ $# -ne 0 ]]; do
  PREFIX="$1"; shift

  TESTED+=("${PREFIX}")

  PASSED_TAG="${PREFIX}.passed"

  rm -f "${BINDIR}/${PASSED_TAG}"

  # Information to be recorded
  CIRCLE_VALUE_PASSED=FALSE

  cat > "${BINDIR}/${PREFIX}.log" <(
    exec 2>&1

    # Exit immediately if any command fails
    set -e
    # Show commands
    set -x

    # Run nnpkg_test in remote machine
    if [ ! -d "${PREFIX}" ] ; then
    PREFIX=${PREFIX}.opt ;
    fi
    scp -r "${PREFIX}/" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_WORKDIR}/${PREFIX}/"
    ssh "${REMOTE_USER}@${REMOTE_IP}" "cd ${REMOTE_WORKDIR}; ./Product/out/test/onert-test nnpkg-test ${PREFIX}"

    if [[ $? -eq 0 ]]; then
      touch "${BINDIR}/${PASSED_TAG}"
    fi
  )

  if [[ -f "${BINDIR}/${PASSED_TAG}" ]]; then
    PASSED+=("$PREFIX")
    CIRCLE_VALUE_PASSED=TRUE
  else
    FAILED+=("$PREFIX")
    CIRCLE_VALUE_PASSED=FALSE
  fi

  echo "${PREFIX}, ${CIRCLE_VALUE_PASSED}" >> ${RESULT_CSV}
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
