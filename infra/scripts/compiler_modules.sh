#!/bin/bash

# NOTE this file is sourced from, for the purpose of
# - configure_compiler_coverage.sh: to get test coverage for release criteria

# Don't run this script
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && echo "Please don't execute ${BASH_SOURCE[0]}, source it" && return

DEBUG_BUILD_ITEMS="angkor;cwrap;pepper-str;pepper-strcast;pp"
DEBUG_BUILD_ITEMS+=";oops;pepper-assert;pepper-csv2vec"
DEBUG_BUILD_ITEMS+=";hermes;hermes-std"
DEBUG_BUILD_ITEMS+=";loco;locop;locomotiv;logo-core;logo"
DEBUG_BUILD_ITEMS+=";foder;crew;souschef;arser;vconone"
DEBUG_BUILD_ITEMS+=";safemain;mio-circle05;mio-tflite280;mio-circle06;mio-tflite2121;dio-hdf5"
DEBUG_BUILD_ITEMS+=";luci-compute"
DEBUG_BUILD_ITEMS+=";tflite2circle"
DEBUG_BUILD_ITEMS+=";luci"
DEBUG_BUILD_ITEMS+=";luci-interpreter"
DEBUG_BUILD_ITEMS+=";luci-eval-driver;luci-pass-value-test;luci-value-test"
DEBUG_BUILD_ITEMS+=";circle2circle;record-minmax;circle-quantizer"
DEBUG_BUILD_ITEMS+=";circle-eval-diff"
DEBUG_BUILD_ITEMS+=";circle-partitioner;circle-part-driver;circle-operator"
DEBUG_BUILD_ITEMS+=";circle-verify"
DEBUG_BUILD_ITEMS+=";circle-tensordump;circle-opselector"
DEBUG_BUILD_ITEMS+=";tflchef;circlechef"
DEBUG_BUILD_ITEMS+=";common-artifacts"
DEBUG_BUILD_ITEMS+=";circle2circle-dredd-recipe-test"
DEBUG_BUILD_ITEMS+=";record-minmax-conversion-test"
DEBUG_BUILD_ITEMS+=";tf2tfliteV2;tf2tfliteV2-conversion-test"
DEBUG_BUILD_ITEMS+=";tflite2circle-conversion-test"
DEBUG_BUILD_ITEMS+=";pota-quantization-value-test;pics"
DEBUG_BUILD_ITEMS+=";circle-part-value-test"
DEBUG_BUILD_ITEMS+=";circle-quantizer-dredd-recipe-test"
DEBUG_BUILD_ITEMS+=";circle-operator-test"
DEBUG_BUILD_ITEMS+=";circle-interpreter;circle-interpreter-test"
DEBUG_BUILD_ITEMS+=";dalgona;dalgona-test"
DEBUG_BUILD_ITEMS+=";visq"
DEBUG_BUILD_ITEMS+=";circle-mpqsolver"
