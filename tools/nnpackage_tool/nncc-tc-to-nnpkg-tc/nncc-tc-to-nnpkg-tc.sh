#!/bin/bash

set -eu

progname=$(basename "${BASH_SOURCE[0]}")
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
model2nnpkg=${model2nnpkg:-"$script_dir"/../model2nnpkg/model2nnpkg.sh}
# Need to install nncc package & set path to tf2nnpkg
tf2nnpkg=$(which tf2nnpkg)

indir="."
outdir="."

usage() {
  echo "Usage: $progname [options] nncc_tc_name"
  echo "Convert nncc testcase to nnpackage testcase."
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -i   set input directory (default=$indir)"
  echo "    -o   set nnpackage testcase output directory (default=$outdir)"
  echo ""
  echo "Env:"
  echo "   model2nnpkg    path to model2nnpkg tool (default={this_script_home}/../model2nnpkg)"
  echo ""
  echo "Examples:"
  echo "    $progname -i build/compiler/tf2tflite UNIT_Add_000"
  echo "      => create nnpackage testcase in $outdir/ from build/compiler/tf2tflite/UNIT_Add_000.*"
  echo "    $progname -o out UNIT_Add_000"
  echo "      => create nnpackage testcase in out/ using $indir/UNIT_Add_000.*"
  exit 1
}

if [ $# -eq 0 ]; then
  echo "For help, type $progname -h"
  exit 1
fi

while getopts "hi:o:" OPTION; do
case "${OPTION}" in
    h) usage;;
    i) indir=$OPTARG;;
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

tcname=$1

supported_model_types="
pb
circle
tflite
"

model_type=""
tf_intf_version=""

for ext in $supported_model_types; do
  [ -e "$indir/$tcname"."$ext" ] && model_type=$ext
done;

if [[ "$model_type" == "" ]]; then
  echo "error: No modelfile is found in $indir/$tcname*"
  exit 1
fi

if [[ "$model_type" == "pb" ]]; then
  [ -f "$indir/$tcname"."v2" ] && tf_intf_version="--v2"
  $tf2nnpkg --info "$indir/$tcname".info --graphdef "$indir/$tcname"."$model_type" \
  "$tf_intf_version" -o "$outdir"
else
  $model2nnpkg -o "$outdir" "$indir/$tcname"."$model_type"
fi

extensions="
expected.h5
input.h5
"

destdir="$outdir/$tcname/metadata/tc"
mkdir -p "$destdir"
for ext in $extensions; do
  cp "$indir/$tcname.$ext" "$destdir/$ext"
done;

