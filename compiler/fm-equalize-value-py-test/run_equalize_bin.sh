#!/bin/bash

# TODO add check arguments

VIRTUAL_ENV=$1; shift
FM_EQUALIZE_BIN=$1; shift
OPT_CIRCLE_OUTPUT_PATH=$1; shift
AFTER_CIRCLE_OUTPUT_PATH=$1; shift
AFTER_CIRCLE_PATTERN_PATH=$1; shift
FME_DETECT_BIN=$1; shift
DALGONA_BIN=$1; shift
FME_APPLY_BIN=$1; shift

source ${VIRTUAL_ENV}/bin/activate

python ${FM_EQUALIZE_BIN} \
  -i ${OPT_CIRCLE_OUTPUT_PATH} \
  -o ${AFTER_CIRCLE_OUTPUT_PATH} \
  -f ${AFTER_CIRCLE_PATTERN_PATH} \
  --fme_detect ${FME_DETECT_BIN} \
  --dalgona ${DALGONA_BIN} \
  --fme_apply ${FME_APPLY_BIN}

deactivate
