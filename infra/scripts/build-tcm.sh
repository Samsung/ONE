#!/bin/bash
#
# STEP 1
#   Download latest TCM tool from 
#   https://github.sec.samsung.net/RS-TCM/tca-standalone/releases/download/1.0.2/tca-standalone-1.0.2.jar
#
# STEP 2
#   Create symbolic link `./src` for source directory to be analyzed which has `.ahub` configuration.
#
# STEP 3
#   run this script in `build-tcm.sh [test_target]` format.
#   ex) $ build_tcm.sh                # to analyze both NN Runtime and NN Compiler
#   ex) $ build_tcm.sh NN_Runtime     # to analyze NN Runtime only
#   ex) $ build_tcm.sh NN_Compiler    # to analyze NN Compiler only
#
# See the following link for additional details.
#   https://github.sec.samsung.net/RS-TCM/tca-standalone/wiki/Tutorials-CPP-Gtest
#

echo ${PROJECT_DIR:=${PWD}}

java -jar $PROJECT_DIR/tca-standalone-1.0.2.jar \
  --outdir=$PROJECT_DIR/tcm-output \
  --config=$PROJECT_DIR/src/.ahub/tcchecker-tca/config.yaml \
  --local=$PROJECT_DIR/src \
  --logfile=$PROJECT_DIR/tcm-output/tcm.log \
  --debug
  $@
