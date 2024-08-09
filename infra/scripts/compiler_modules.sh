#!/bin/bash

# NOTE this file is sourced from, for the purpose of
# - configure_compiler_coverage.sh (DEBUG_BUILD_ITEMS): to get test coverage for release criteria
# - configure_collect_nnpkgs.sh (NNPKG_RES_ITEMS): to collect nnpkg resources for nnpackage test

# Don't run this script
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && echo "Please don't execute ${BASH_SOURCE[0]}, source it" && return

DEBUG_BUILD_ITEMS="angkor;cwrap;pepper-str;pepper-strcast;pp"
DEBUG_BUILD_ITEMS+=";oops;pepper-assert;pepper-csv2vec"
DEBUG_BUILD_ITEMS+=";hermes;hermes-std"
DEBUG_BUILD_ITEMS+=";loco;locop;locomotiv;logo-core;logo"
DEBUG_BUILD_ITEMS+=";foder;crew;souschef;arser;vconone"
DEBUG_BUILD_ITEMS+=";safemain;mio-circle09;mio-tflite2121;dio-hdf5"
DEBUG_BUILD_ITEMS+=";luci-compute"
DEBUG_BUILD_ITEMS+=";tflite2circle"
DEBUG_BUILD_ITEMS+=";luci"
DEBUG_BUILD_ITEMS+=";luci-interpreter"
DEBUG_BUILD_ITEMS+=";luci-eval-driver;luci-pass-value-py-test;luci-value-py-test"
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
DEBUG_BUILD_ITEMS+=";circle-part-value-py-test"
DEBUG_BUILD_ITEMS+=";circle-quantizer-dredd-recipe-test"
DEBUG_BUILD_ITEMS+=";circle-operator-test"
DEBUG_BUILD_ITEMS+=";circle-interpreter;circle-interpreter-test"
DEBUG_BUILD_ITEMS+=";dalgona;dalgona-test"
DEBUG_BUILD_ITEMS+=";visq"
DEBUG_BUILD_ITEMS+=";circle-mpqsolver"

NNPKG_RES_ITEMS="angkor;cwrap;pepper-str;pepper-strcast;pp"
NNPKG_RES_ITEMS+=";pepper-csv2vec"
NNPKG_RES_ITEMS+=";oops;safemain;foder;crew;arser;vconone"
# Hermes Logging Framework
NNPKG_RES_ITEMS+=";hermes;hermes-std"
# loco IR and related utilities
NNPKG_RES_ITEMS+=";loco;locop;locomotiv;logo-core;logo"
# Compute
NNPKG_RES_ITEMS+=";luci-compute"
# Circle compiler library (.circle -> .circle)
NNPKG_RES_ITEMS+=";luci"
# Flatbuffer I/O
NNPKG_RES_ITEMS+=";mio-tflite2121;mio-circle09"
# Tools
NNPKG_RES_ITEMS+=";tflite2circle;circle2circle;luci-interpreter"
NNPKG_RES_ITEMS+=";souschef;tflchef;circlechef;circle-verify"
# common-artifacts
NNPKG_RES_ITEMS+=";common-artifacts"
