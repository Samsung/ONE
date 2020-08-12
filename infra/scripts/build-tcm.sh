#!/bin/bash
#
# STEP 1
#   Download latest TCM tool from 
#   https://github.sec.samsung.net/RS-TCM/tca-standalone/releases/download/v0.0.8/tca-standalone-0.0.8.jar
#
# STEP 2
#   Create symbolic link `./src` for source directory to be analyzed which has `.ahub` configuration.
#
# STEP 3
#   run this `build-tcm.sh` script.
#
# See the following link for additional details.
#   https://github.sec.samsung.net/RS-TCM/tca-standalone/wiki/Tutorials-CPP-Gtest
#

echo ${PROJECT_DIR:=${PWD}}

java -jar $PROJECT_DIR/tca-standalone-0.0.8.jar \
  --outdir=$PROJECT_DIR/tcm-output \
  --config=$PROJECT_DIR/src/.ahub/tcchecker-tca/config.yaml \
  --local=$PROJECT_DIR/src \
  --logfile=$PROJECT_DIR/tcm-output/tcm.log \
  --debug
