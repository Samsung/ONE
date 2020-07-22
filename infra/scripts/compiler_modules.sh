#!/bin/bash

# Don't run this script
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && echo "Please don't execute ${BASH_SOURCE[0]}, source it" && return

DEBUG_BUILD_ITEMS="angkor;cwrap;pepper-str;pepper-strcast;pp;stdex"
DEBUG_BUILD_ITEMS+=";oops;pepper-assert"
DEBUG_BUILD_ITEMS+=";hermes;hermes-std"
DEBUG_BUILD_ITEMS+=";loco;locop;locomotiv;logo-core;logo"
DEBUG_BUILD_ITEMS+=";foder;souschef;arser;vconone"
DEBUG_BUILD_ITEMS+=";safemain;mio-circle;mio-tflite"
DEBUG_BUILD_ITEMS+=";tflite2circle"
DEBUG_BUILD_ITEMS+=";luci"
DEBUG_BUILD_ITEMS+=";luci-interpreter"
DEBUG_BUILD_ITEMS+=";luci-value-test"
DEBUG_BUILD_ITEMS+=";circle2circle;record-minmax;circle-quantizer"
DEBUG_BUILD_ITEMS+=";circle-verify"
DEBUG_BUILD_ITEMS+=";tflchef;circlechef"
DEBUG_BUILD_ITEMS+=";common-artifacts"
DEBUG_BUILD_ITEMS+=";circle2circle-dredd-recipe-test"
DEBUG_BUILD_ITEMS+=";record-minmax-conversion-test"
DEBUG_BUILD_ITEMS+=";tf2tfliteV2;tf2tfliteV2-conversion-test"
DEBUG_BUILD_ITEMS+=";tflite2circle-conversion-test"
