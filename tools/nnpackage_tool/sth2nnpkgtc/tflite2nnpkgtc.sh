#!/bin/bash

set -u

progname=$(basename "${BASH_SOURCE[0]}")
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
nnfw_root="$( cd "${script_dir%*/*/*/*}" >/dev/null && pwd )"
outdir="."

usage() {
  echo "Usage: $progname [options] tflite"
  echo "Convert tflite to nnpkg-tc"
  echo ""
  echo "Returns"
  echo "     0       success"
  echo "  non-zero   failure"
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -o   set output directory (default=$outdir)"
  echo ""
  echo "Examples:"
  echo "    $progname your.tflite"
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

tflite_basename=$(basename "$1")
name=${tflite_basename%.*}

tools/nnpackage_tool/gen_golden/gen_golden.py $1
tools/nnpackage_tool/model2nnpkg/model2nnpkg.py -o ${outdir} -m $1
mkdir -p ${outdir}/$name/metadata/tc
mv {input,expected}.h5 ${outdir}/$name/metadata/tc/
cp $1 ${outdir}/$name/
