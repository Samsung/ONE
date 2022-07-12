#!/bin/bash

#
# Copy model files into ./models directory
# Then create ./models.lst
#

set -e # exit on error
#set -x # trace

#MODEL_DIR=/tmp/one-init-validate
CUR_DIR=$(pwd)
MODEL_DIR=${CUR_DIR}/models

if [[ ! -d ${MODEL_DIR} ]]; then
  echo "${MODEL_DIR} does not exist."
  exit 1
fi

# TODO Support pb

MODEL_URL_FILE='./models.lst'

while read LINE; do

  # skip comment
  if [[ $LINE == '#'* || $LINE == '' ]]; then continue; fi

  RELATIVE_MODEL_PATH=$(echo "${LINE}" | cut -d' ' -f1)
  ONE_INIT_OPTION=$(echo "${LINE}" | cut -d' ' -f2-)
  if [[ "$ONE_INIT_OPTION" == "NONE" ]]; then
    ONE_INIT_OPTION=""
  fi

  echo "#================================================="
  echo "# Compiling ${RELATIVE_MODEL_PATH}"
  echo "#================================================="

  MODEL_PATH=${CUR_DIR}/${RELATIVE_MODEL_PATH}
  MODEL_NAME=$(basename ${MODEL_PATH})
  LOWER_CASE_MODEL_NAME=$(echo $MODEL_NAME | tr '[:upper:]' '[:lower:]')

  if [[ $LOWER_CASE_MODEL_NAME == *.tflite || $LOWER_CASE_MODEL_NAME == *.onnx ]]
  then
    EXT="${LOWER_CASE_MODEL_NAME##*.}"
    cp -f ${MODEL_PATH} ${MODEL_DIR}/model.${EXT}

    # Generate cfg
    python ../one-init -i ${MODEL_DIR}/model.${EXT} -o ${MODEL_DIR}/model.cfg -b tv2 \
                      ${ONE_INIT_OPTION}

    echo
    echo "### onecc ${MODEL_NAME} "

    pushd ${MODEL_DIR}
    CMD="onecc -V -C ${MODEL_DIR}/model_out.cfg"
    echo "$CMD"
    command $CMD
    if [[ $? != 0 ]]; then
      echo
      echo "[ERROR] Compilation failed."
      exit 1
    fi

    echo
    echo "### tvn compiled from ${MODEL_NAME} ---------"
    ls -l $MODEL_DIR/*.tvn

    rm -f $MODEL_DIR/model.${EXT}
    rm -f $MODEL_DIR/*.cfg
    rm -f $MODEL_DIR/*.{circle,log}  # -f does not make error when file does not exist
    rm -f $MODEL_DIR/*.{tv*}

    popd

  else
    echo "Unknown extension: ${EXT}"
    exit 1
  fi

done < ${MODEL_URL_FILE}
