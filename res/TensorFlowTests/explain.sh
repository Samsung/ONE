#!/bin/bash

TESTCASE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

### Parse Command-line Arguments
SHOW_PATH=0

while true; do
  case "$1" in
    -p | --path)
      SHOW_PATH=1;
      shift
      ;;
    *)
      break
      ;;
  esac
done

### Generate Report
(
  # Print HEADER
  echo -n "NAME"
  echo -n ",SUMMARY"
  if [[ ${SHOW_PATH} -ne 0 ]]; then
    echo -n ",PATH"
  fi
  echo

  # Print ROW(s)
  for PREFIX in $(cd "${TESTCASE_REPO}"; ls */test.info | xargs -i dirname {} | sort); do
    TESTCASE_DIR="${TESTCASE_REPO}/${PREFIX}"
    TESTCASE_MANIFEST_FILE="${TESTCASE_DIR}/test.manifest"
    TESTCASE_GRAPHDEF_FILE="${TESTCASE_DIR}/test.pbtxt"

    echo -n "${PREFIX}"
    if [[ -f "${TESTCASE_MANIFEST_FILE}" ]]; then
      echo -n ",$(cat "${TESTCASE_MANIFEST_FILE}" | grep '^SUMMARY: ' | head -n1 | sed 's/^SUMMARY://g'  | xargs)"
    else
      echo -n ",-"
    fi
    if [[ ${SHOW_PATH} -ne 0 ]]; then
      echo -n ",${TESTCASE_GRAPHDEF_FILE}"
    fi
    echo
  done
) | column -t -s ,
