#!/bin/bash

usage()
{
  echo "$0 <options>"
  echo "Options"
  echo "--onert_run : specific onert_run path"
  echo "--tflite_run : specific tflite_run path"
  echo "--dir : the dir path of models"
  echo "--list : the model list"
  echo "--out  : the file name of out results"
  echo "--tv   : for tv"
  exit 1
}

scripts_dir="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"
nnfw_dir="${scripts_dir}/../.."
onert_run="${nnfw_dir}/Product/out/bin/onert_run"
tflite_run="${nnfw_dir}/Product/out/bin/tflite_run"
base_name="$(basename $0)"
base_name="${base_name%.*}"
outfile="${base_name}_result.txt"
dir=""
list="${scripts_dir}/list/${base_name}_model_list.txt"
tv_on="false"

for i in "$@"
do
case $i in
  --onert_run=*)
    onert_run="${i#*=}"
    ;;
  --tflite_run=*)
    tflite_run="${i#*=}"
    ;;
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
  *)
    ;;
esac
shift
done

if ! [ -f ${onert_run} ]; then
  echo "onert_run file does not exists."
  usage
fi

if ! [ -f ${tflite_run} ]; then
  echo "tflite_run file does not exists."
  usage
fi

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

  TFLITE_CMD="LD_LIBRARY_PATH=./Product/out/lib THREAD=3 ${tflite_run} -r 10 -m 1 -p 1"
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

${scripts_dir}/merge_result_of_benchmark_nnpkg.py -i . -o . -l ${list}
