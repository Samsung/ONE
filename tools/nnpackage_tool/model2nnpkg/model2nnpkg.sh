#!/bin/bash

set -eu

progname=$(basename "${BASH_SOURCE[0]}")
outdir="."
name=""
config=""
config_src=""
models_str=""
types_str=""

usage() {
  echo "Usage: $progname [options] modelfile..."
  echo "Convert modelfile (tflite, circle or tvn) to nnpackage."
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -o   set nnpackage output directory (default=$outdir)"
  echo "    -p   set nnpackage output name (default=[modelfile name])"
  echo "    -c   provide configuration file"
  echo ""
  echo "Examples:"
  echo "    $progname add.tflite                  => create nnpackage 'add' in $outdir/"
  echo "    $progname -o out add.tflite           => create nnpackage 'add' in out/"
  echo "    $progname -o out -p addpkg add.tflite => create nnpackage 'addpkg' in out/"
  echo "    $progname -c add.cfg add.tflite       => create nnpackage 'add' with add.cfg"
  exit 1
}

if [ $# -eq 0 ]; then
  >&2 echo "For help, type $progname -h"
  exit 1
fi

while getopts "ho:p:c:" OPTION; do
case "${OPTION}" in
    h) usage;;
    o) outdir=$OPTARG;;
    p) name=$OPTARG;;
    c) config_src=$OPTARG;;
    ?) exit 1;;
esac
done

shift $((OPTIND-1))

if [ $# -eq 0 ]; then
  >&2 echo "error: wrong argument (no model argument)."
  >&2 echo "For help, type $progname -h"
  exit 1
fi

for modelpath in "$@"
do
  modelfile=$(basename "$modelpath")

  if [[ "$modelfile" != *.* ]]; then
    >&2 echo "error: modelfile does not have extension."
    >&2 echo "Please provide extension so that $progname can identify what type of model you use."
    exit 1
  fi

  if [ ! -e $modelpath ]; then
    >&2 echo "error: "$modelpath" does not exist."
    exit 1
  fi

  models_str="$models_str$delim\"$modelfile\""
  types_str="$types_str$delim\"${modelfile##*.}\""
  delim=", "
done

if [ -z "$name" ]; then
  first_modelfile=$(basename "$1")
  name=${first_modelfile%.*}
fi

echo "$progname: Generating nnpackage "$name" in "$outdir""
mkdir -p "$outdir"/"$name"/metadata

if [ -s "$config_src" ]; then
  config=$(basename "$config_src")
  cp "$config_src" "$outdir/$name/metadata/$config"
fi

cat > "$outdir"/"$name"/metadata/MANIFEST <<-EOF
{
  "major-version" : "1",
  "minor-version" : "2",
  "patch-version" : "0",
  "configs"     : [ "$config" ],
  "models"      : [ $models_str ],
  "model-types" : [ $types_str ]
}
EOF

for modelpath in "$@"
do
  cp "$modelpath" "$outdir"/"$name"
done
