#!/bin/bash

MY_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source $MY_PATH/common.sh

# Caution: DO NOT USE "pipefail"
#          We should run test all nnpackages

onert_run="$INSTALL_PATH/bin/onert_run"
tflite_run="$INSTALL_PATH/bin/tflite_run"
base_name="$(basename $0)"
base_name="${base_name%.*}"
outfile="${base_name}_result.txt"
dir=""
list="$INSTALL_PATH/test/list/benchmark_nnpkg_model_list.txt"
tv_on="false"

function usage()
{
  echo "Usage: ${BASH_SOURCE[0]} [OPTIONS]"
  echo "Options"
  echo "    --dir=PATH    : the dir path of models"
  echo "    --list=FILE   : the model list (default: $list)"
  echo "    --out=FILE    : the file name of out results (default: $outfile)"
  echo "    --tv          : for tv"
  echo "    --help        : display this help message and exit"
  exit 1
}

for i in "$@"
do
  case $i in
    --out=*)
      outfile="${i#*=}"
      ;;
    --dir=*)
      dir="${i#*=}"
      ;;
    --list=*)
      list="${i#*=}"
      ;;
    --tv)
      tv_on="true"
      ;;
    --help)
      usage
      exit 1
      ;;
    *)
      ;;
  esac
  shift
done

if ! [ -f ${list} ]; then
  echo "model list file does not exists."
  usage
fi

if [ -z ${dir} ]; then
  echo "dir is empty."
  usage
fi

if ! [ -d ${dir} ]; then
  echo "dir does not exists."
  usage
fi

if [ -z ${outfile} ]; then
  echo "outfile is empty."
  usage
fi

if ! [ -f ${outfile} ]; then
  touch ${outfile}
fi

# get lists
model_lists=()
for model_name in `cat $list`; do
  model_lists+=($model_name)
done

# run
for i in "${model_lists[@]}"; do
  echo "${i} result" | tee -a ${outfile}

  CMD="${onert_run} -r 10 -m 1 -p 1"
  if [ "$tv_on" == "true" ]; then
    ${CMD}="${CMD} -g 1"
  fi
  CMD="${CMD} ${dir}/${i} 2>&1 >> ${outfile}"

  # cpu
  CPU_CMD="BACKENDS=cpu ${CMD}"
  echo "${CPU_CMD}"
  echo "" >> ${outfile}
  echo "onert cpu" >> ${outfile}
  eval "${CPU_CMD}"

  sleep 10 # for avoiding cpu overheated

  # acl_neon
  NEON_CMD="BACKENDS=acl_neon ${CMD}"
  echo "${NEON_CMD}"
  echo "" >> ${outfile}
  echo "onert acl_neon" >> ${outfile}
  eval "${NEON_CMD}"

  sleep 10 # for avoiding cpu overheated

  # acl_cl
  CL_CMD="BACKENDS=acl_cl ${CMD}"
  echo "${CL_CMD}"
  echo "" >> ${outfile}
  echo "onert acl_cl" >> ${outfile}
  eval "${CL_CMD}"

  echo "" >> ${outfile}

  TFLITE_CMD="THREAD=3 ${tflite_run} -r 10 -m 1 -p 1"
  if [ "$tv_on" == "true" ]; then
    TFLITE_CMD="${TFLITE_CMD} -g 1"
  fi
  TFLITE_CMD="${TFLITE_CMD} ${dir}/${i}/${i}.tflite >> ${outfile}"

  echo "TfLite + CPU" >> ${outfile}
  echo "${TFLITE_CMD}"
  eval "${TFLITE_CMD}"

  echo "" >> ${outfile}

  sleep 20 # for avoiding cpu overheated
done # ${model_lists}

python3 $MY_PATH/merge_result_of_benchmark_nnpkg.py -i . -o . -l ${list}
