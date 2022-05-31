#!/bin/bash

set -eu

progname=$(basename "${BASH_SOURCE[0]}")
outdir="."
name=""
configs_src=()
models_src=()
configs_str=""
models_str=""
types_str=""

usage() {
  echo "Usage: $progname [options]"
  echo "Convert modelfile (tflite, circle or tvn) to nnpackage."
  echo ""
  echo "Options:"
  echo "    -h   show this help"
  echo "    -o   set nnpackage output directory (default=$outdir)"
  echo "    -p   set nnpackage output name (default=[1st modelfile name])"
  echo "    -c   provide configuration files"
  echo "    -m   provide model files"
  echo ""
  echo "         (Will be deprecated: if there is one remain parameter, that is model file)"
  echo ""
  echo "Examples:"
  echo "    $progname -m add.tflite                           => create nnpackage 'add' in $outdir/"
  echo "    $progname -o out -m add.tflite                    => create nnpackage 'add' in out/"
  echo "    $progname -o out -p addpkg -m add.tflite          => create nnpackage 'addpkg' in out/"
  echo "    $progname -c add.cfg -m add.tflite                => create nnpackage 'add' with add.cfg"
  echo "    $progname -o out -p addpkg -m a1.tflite a2.tflite => create nnpackage 'addpkg' with models a1.tflite and a2.tflite in out/"
  echo ""
  echo "(Will be deprecated: if there is one remain parameter, that is model file)"
  exit 1
}

if [ $# -eq 0 ]; then
  >&2 echo "For help, type $progname -h"
  exit 1
fi

while getopts "ho:p:c:m:" OPTION; do
  case "${OPTION}" in
    h) usage;;
    o) outdir=$OPTARG;;
    p) name=$OPTARG;;
    c)
      configs_src=($OPTARG)
      until [[ $OPTIND -gt $# ]] || [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [ -z $(eval "echo \${$OPTIND}") ]; do
        if [[ $OPTIND -eq $# ]] && [[ ${#models_src[@]} -eq 0 ]]; then
          # Backward compatibility (will be deprecated)
          # The last remain parameter is model if there is no option "-m"
          models_src=($(eval "echo \${$OPTIND}"))
        else
          configs_src+=($(eval "echo \${$OPTIND}"))
        fi
        OPTIND=$((OPTIND + 1))
      done
      ;;
    m)
      models_src=($OPTARG)
      until [[ $OPTIND -gt $# ]] || [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [ -z $(eval "echo \${$OPTIND}") ]; do
        models_src+=($(eval "echo \${$OPTIND}"))
        OPTIND=$((OPTIND + 1))
      done
      ;;
    ?) exit 1;;
  esac
done

shift $((OPTIND-1))

# Backward compatibility (will be deprecated)
# The last remain parameter is model if there is no option "-m"
if [ $# -eq 1 ] && [ ${#models_src[@]} -eq 0 ]; then
  models_src=($1)
  shift 1
fi

if [ $# -ne 0 ]; then
  >&2 echo "error: wrong argument (too many arguments)."
  >&2 echo "For help, type $progname -h"
  exit 1
fi

if [[ ${#configs_src[@]} -ne 0 ]] && [[ ${#configs_src[@]} -ne ${#models_src[@]} ]]; then
  >&2 echo "error: when config file is provided, # of config file should be same with modelfile"
  >&2 echo "Please provide config file for each model file, or don't provide config file."
  exit 1
fi

delim=""
for modelpath in ${models_src[@]}
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

delim=""
for configpath in ${configs_src[@]}
do
  configfile=$(basename "$configpath")

  if [ ! -e $configpath ]; then
    >&2 echo "error: "$configpath" does not exist."
    exit 1
  fi

  configs_str="$configs_str$delim\"$configfile\""
  delim=", "
done

if [ -z "$name" ]; then
  first_modelfile=$(basename "${models_src[0]}")
  name=${first_modelfile%.*}
fi

echo "$progname: Generating nnpackage "$name" in "$outdir""
mkdir -p "$outdir"/"$name"/metadata

cat > "$outdir"/"$name"/metadata/MANIFEST <<-EOF
{
  "major-version" : "1",
  "minor-version" : "2",
  "patch-version" : "0",
  "configs"     : [ $configs_str ],
  "models"      : [ $models_str ],
  "model-types" : [ $types_str ]
}
EOF

for modelpath in ${models_src[@]}
do
  cp "$modelpath" "$outdir"/"$name"
done

for configpath in ${configs_src[@]}
do
  cp "$configpath" "$outdir/$name/metadata"
done
