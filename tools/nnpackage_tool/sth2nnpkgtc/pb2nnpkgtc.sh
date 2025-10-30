#!/bin/bash

set -u

progname=$(basename "${BASH_SOURCE[0]}")
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
nnfw_root="$( cd "${script_dir%*/*/*/*}" >/dev/null && pwd )"
outdir="."

# set path to tools and resources
flatc=${flatc:-"$nnfw_root/build/externals/FLATBUFFERS/build/flatc"}
tflite_schema=${tflite_schema:-"$nnfw_root/externals/tensorflow/tensorflow/lite/schema/schema.fbs"}
circle_schema=${circle_schema:-"$nnfw_root/nnpackage/schema/circle_schema.fbs"}
tensorflow=${tensorflow:-"../tensorflow.git"}

usage() {
  echo "Usage: $progname [options] pb inputs outputs"
  echo "Convert pb to nnpkg-tc"
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
  echo "    $progname your.pb placeholder output"
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

if [ $# -ne 3 ]; then
  echo "error: wrong argument (no argument or too many arguments)."
  echo "For help, type $progname -h"
  exit 1
fi

pb_basename=$(basename "$1")
name=${pb_basename%.*}
inputs=$2
outputs=$3
suffix=${3//\//_}

${script_dir}/pb_select_graph.py $1 $2 $3 $name.$suffix
tflite_convert --output_file=$name.$suffix.tflite --graph_def_file=$name.$suffix.pb --input_arrays=${inputs} --output_arrays=${outputs}
${flatc} --defaults-json --strict-json -t ${tflite_schema} -- $name.$suffix.tflite
node tools/nnpackage_tool/tflite2circle/fuse_instance_norm.js $name.$suffix.json
tools/nnpackage_tool/tflite2circle/tflitejson2circlejson.py $name.$suffix.json.fused > $name.$suffix.json.fused.datalayout
${flatc} -o ./ -b ${circle_schema} $name.$suffix.json.fused.datalayout
mv $name.$suffix.json.fused.circle $name.$suffix.circle
tools/nnpackage_tool/gen_golden/gen_golden.py $name.$suffix.pb
tools/nnpackage_tool/model2nnpkg/model2nnpkg.py -o ${outdir} -m $name.$suffix.circle
mkdir -p ${outdir}/$name.$suffix/metadata/tc
mv {input,expected}.h5 ${outdir}/$name.$suffix/metadata/tc/
mv $name.$suffix.{pb,tflite} ${outdir}/$name.$suffix/
rm $name.$suffix.circle $name.$suffix.json*
