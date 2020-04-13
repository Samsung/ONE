#!/bin/bash

TESTCASE_REPO="$1"; shift
TFKIT_PATH="$1"; shift
NNKIT_RUN_PATH="$1"; shift
NNKIT_TF_BACKEND_PATH="$1"; shift

echo "-- Found TensorFlow testcases: '${TESTCASE_REPO}'"
echo "-- Found tfkit: '${TFKIT_PATH}'"
echo "-- Found nnkit-run: '${NNKIT_RUN_PATH}'"
echo "-- Found nnkit TensorFlow backend: '${NNKIT_TF_BACKEND_PATH}'"

EXITCODE=0

PASSED=()
FAILED=()
SKIPPED=()

for PREFIX in $(cd "${TESTCASE_REPO}"; ls */test.info | xargs -i dirname {} | sort); do
  TESTCASE_DIR="${TESTCASE_REPO}/${PREFIX}"

  if [[ ! -f "${TESTCASE_DIR}/customop.conf" ]]; then
    PASSED_TAG="${PREFIX}.passed"

    cat > "${PREFIX}.log" <(
      exec 2>&1

      rm -f "${PASSED_TAG}"

      set -ex
      # Create a pb model
      "${TFKIT_PATH}" encode "${TESTCASE_DIR}/test.pbtxt" "${PREFIX}.pb"

      # Do inference
      "${NNKIT_RUN_PATH}" \
        --backend "${NNKIT_TF_BACKEND_PATH}" \
        --backend-arg "${PREFIX}.pb" \
        --backend-arg "${TESTCASE_DIR}/test.info"
      set +ex

      touch "${PASSED_TAG}"
    )

    if [[ ! -f "${PASSED_TAG}" ]]; then
      FAILED+=("$PREFIX")
      RESULT="FAIL"
    else
      PASSED+=("$PREFIX")
      RESULT="PASS"
    fi
  else
    SKIPPED+=("$PREFIX")
    RESULT="SKIP"
  fi

  echo "Check '${PREFIX}' testcase - ${RESULT}"
done

echo
echo "PASSED: ${#PASSED[@]}, FAILED: ${#FAILED[@]}, SKIPPED: ${#SKIPPED[@]}"

if [[ ${#FAILED[@]} -ne 0 ]]; then
  echo
  echo "FAILED"
  for TEST in "${FAILED[@]}"
  do
    echo "- ${TEST}"
  done
  exit 255
fi

exit 0
