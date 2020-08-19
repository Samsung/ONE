#!/bin/bash

set -u

progname=$(basename "${BASH_SOURCE[0]}")
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
nnfw_root="$( cd "${script_dir%*/*/*/*}" && pwd )"
outdir="."
flatc=${flatc:-"$nnfw_root/build/externals/FLATBUFFERS/build/flatc"}
tflite_schema=${tflite_schema:-"$nnfw_root/externals/TENSORFLOW-1.13.1/tensorflow/lite/schema/schema.fbs"}
circle_schema=${circle_schema:-"$nnfw_root/nnpackage/schema/circle_schema.fbs"}

if ! [ -x "$flatc" ]; then
  echo "Please make sure `flatc` is in path."
  exit 2
fi

if ! { [ -e "$tflite_schema" ] && [ -e "$circle_schema" ]; }; then
  echo "Please make sure that the `*.fbs` paths are set properly."
  exit 3
fi

usage() {
  echo "Usage: $progname [options] tflite"
  echo "Convert tflite to circle"
  echo ""
  echo "Returns"
  echo "     0       success"
  echo "  non-zero   failure"
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -o   set output directory (default=$outdir)"
  echo ""
  echo "Environment variables:"
  echo "   flatc           path to flatc"
  echo "                   (default=./build/externals/FLATBUFFERS/build/flatc)"
  echo "   tflite_schema   path to tflite schema (i.e. schema.fbs)"
  echo "                   (default=./externals/TENSORFLOW-1.12/tensorflow/contrib/lite/schema/schema.fbs)"
  echo "   circle_schema   path to circle schema"
  echo "                   (default=./nnpackage/schema/circle_schema.fbs)"
  echo ""
  echo "Examples:"
  echo "    $progname Add_000.tflite         => convert Add_000.tflite into Add_000.circle"
  echo "    $progname -o my/circles Add_000  => convert Add_000.tflite into my/circles/Add_000.circle"
  exit 1
}

if [ $# -eq 0 ]; then
  echo "For help, type $progname -h"
  exit 1
fi

while getopts "ho:" OPTION; do
case "${OPTION}" in
    h) usage;;
    o) outdir=$OPTARG;;
    ?) exit 1;;
esac
done

shift $((OPTIND-1))

if [ $# -ne 1 ]; then
  echo "error: wrong argument (no argument or too many arguments)."
  echo "For help, type $progname -h"
  exit 1
fi

tflite_base=$(basename "$1")
name=${tflite_base%.*}

# convert

mkdir -p "${outdir}"
${flatc} -o ${outdir} --strict-json -t ${tflite_schema} -- $1
${script_dir}/tflitejson2circlejson.py "${outdir}/${name}.json" > "${outdir}/${name}.circle"
${flatc} -o ${outdir} -b ${circle_schema} "${outdir}/${name}.circle"
rm -f ${outdir}/${name}.json
