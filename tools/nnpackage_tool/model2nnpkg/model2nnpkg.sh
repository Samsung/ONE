#!/bin/bash

set -eu

progname=$(basename "${BASH_SOURCE[0]}")
outdir="."

usage() {
  echo "Usage: $progname [options] modelfile"
  echo "Convert modelfile (either tflite or circle) to nnpackage."
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -o   set nnpackage output directory (default=$outdir)"
  echo ""
  echo "Examples:"
  echo "    $progname add.tflite        => create nnpackage in $outdir/"
  echo "    $progname -o out add.tflite => create nnpackage in out/"
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

modelfile=$(basename "$1")

if [[ "$modelfile" != *.* ]]; then
  echo "error: modelfile does not have extension."
  echo "Please provide extension so that $progname can identify what type of model you use."
  exit 1
fi

if [ ! -e $1 ]; then
  echo "error: "$1" does not exist."
  exit 1
fi

name=${modelfile%.*}
extension=${modelfile##*.}

echo "Generating nnpackage "$name" in "$outdir""
mkdir -p "$outdir"/"$name"/metadata
cat > "$outdir"/"$name"/metadata/MANIFEST <<-EOF
{
  "major-version" : "1",
  "minor-version" : "0",
  "patch-version" : "0",
  "models"      : [ "$name.$extension" ],
  "model-types" : [ "$extension" ]
}
EOF
cp "$1" "$outdir"/"$name"
