#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

set -eo pipefail

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

source ${CURRENT_PATH}/compiler_modules.sh

NNCC_CFG_OPTION=" -DCMAKE_BUILD_TYPE=Release"
NNCC_CFG_STRICT=" -DENABLE_STRICT_BUILD=ON"
NNCC_COV_DEBUG=" -DBUILD_WHITELIST=$NNPKG_RES_ITEMS"

if [ $# -ne 0 ]; then
	echo "Additional cmake configuration: $@"
fi

# Reset whitelist to build all
./nncc configure \
	$NNCC_CFG_OPTION $NNCC_COV_DEBUG $NNCC_CFG_STRICT \
	"$@"
