#!/bin/bash

[[ "${BASH_SOURCE[0]}" != "${0}" ]] && echo "Please don't source ${BASH_SOURCE[0]}, execute it" && return

set -eo pipefail

CURRENT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$CURRENT_PATH/../../"

DEBUG_BUILD_ITEMS="angkor;cwrap;pepper-str;pepper-strcast;pp;stdex"
DEBUG_BUILD_ITEMS+=";oops;pepper-assert"
DEBUG_BUILD_ITEMS+=";hermes;hermes-std"
DEBUG_BUILD_ITEMS+=";loco;locop;locomotiv;logo-core;logo"
DEBUG_BUILD_ITEMS+=";foder"
DEBUG_BUILD_ITEMS+=";safemain;mio-circle;mio-tflite"
DEBUG_BUILD_ITEMS+=";tflite2circle"
DEBUG_BUILD_ITEMS+=";luci"
DEBUG_BUILD_ITEMS+=";luci-interpreter"
DEBUG_BUILD_ITEMS+=";luci-value-test"
DEBUG_BUILD_ITEMS+=";circle2circle;record-minmax;circle-quantizer"
DEBUG_BUILD_ITEMS+=";circle-verify"
DEBUG_BUILD_ITEMS+=";tflchef;circlechef"
DEBUG_BUILD_ITEMS+=";circle2circle-dredd-recipe-test"
DEBUG_BUILD_ITEMS+=";record-minmax-conversion-test"
DEBUG_BUILD_ITEMS+=";tf2tfliteV2-conversion-test"
DEBUG_BUILD_ITEMS+=";tflite2circle-conversion-test"

NNCC_CFG_OPTION=" -DCMAKE_BUILD_TYPE=Debug"
NNCC_CFG_STRICT=" -DENABLE_STRICT_BUILD=ON"
NNCC_COV_DEBUG=" -DBUILD_WHITELIST=$DEBUG_BUILD_ITEMS"

if [ $# -ne 0 ]; then
	echo "Additional cmake configuration: $@"
fi

./nncc configure \
	$NNCC_CFG_OPTION $NNCC_COV_DEBUG $NNCC_CFG_STRICT \
	-DENABLE_COVERAGE=ON "$@"
