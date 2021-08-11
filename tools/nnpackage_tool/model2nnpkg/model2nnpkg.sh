#!/bin/bash

set -eu

progname=$(basename "${BASH_SOURCE[0]}")
outdir="."
name=""
config=""
config_src=""

usage() {
  echo "Usage: $progname [options] modelfile"
  echo "Convert modelfile (either tflite or circle) to nnpackage."
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

if [ $# -ne 1 ]; then
  >&2 echo "error: wrong argument (no argument or too many arguments)."
  >&2 echo "For help, type $progname -h"
  exit 1
fi

modelfile=$(basename "$1")

if [[ "$modelfile" != *.* ]]; then
  >&2 echo "error: modelfile does not have extension."
  >&2 echo "Please provide extension so that $progname can identify what type of model you use."
  exit 1
fi

if [ ! -e $1 ]; then
  >&2 echo "error: "$1" does not exist."
  exit 1
fi

if [ -z "$name" ]; then
  name=${modelfile%.*}
fi
extension=${modelfile##*.}

echo "Generating nnpackage "$name" in "$outdir""
mkdir -p "$outdir"/"$name"/metadata

if [ -s "$config_src" ]; then
  config=$(basename "$config_src")
  cp "$config_src" "$outdir/$name/metadata/$config"
fi

cat > "$outdir"/"$name"/metadata/MANIFEST <<-EOF
{
  "major-version" : "1",
  "minor-version" : "1",
  "patch-version" : "0",
  "configs"     : [ "$config" ],
  "models"      : [ "$modelfile" ],
  "model-types" : [ "$extension" ]
}
EOF
cp "$1" "$outdir"/"$name"
