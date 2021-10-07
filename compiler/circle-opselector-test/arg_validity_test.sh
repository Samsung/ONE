#!/bin/bash

# This script test the validity of options
# ex) --by_id "1,2" is valid but --by_id "a,b" is not valid because id parameters must be not negative integer. 
#
# HOW TO USE
#
# ./arg_validity_test.sh <path/to/work_dir> <path/to/venv_dir> <path/to/circle-opselector-dir>
VERIFY_SOURCE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_SCRIPT_PATH="${VERIFY_SOURCE_PATH}/arg_validity_test.py"
ARTIFACTS_DIR="$1"; shift
VIRTUALENV="$1"; shift
COP_DRIVER_PATH="$1"; shift
rm -rf "${ARTIFACTS_DIR}/cop_tmp"
mkdir "${ARTIFACTS_DIR}/cop_tmp"
WORKDIR="${ARTIFACTS_DIR}/cop_tmp"
CIRCLE_ORIGIN="${ARTIFACTS_DIR}/Part_Sqrt_Rsqrt_002.circle"  # Circle file for test, the number of operators is 4

source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "${VERIFY_SCRIPT_PATH}" \
    --opselector "${COP_DRIVER_PATH}" \
    --input "${CIRCLE_ORIGIN}" \
    --output "${WORKDIR}/tmp.circle"

if [[ $? -ne 0 ]]; then
    echo "FAILED"
    exit 255
fi
echo "PASSED"
exit 0
