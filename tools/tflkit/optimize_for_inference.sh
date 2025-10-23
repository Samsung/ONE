#!/bin/bash

usage()
{
  echo "usage : $0"
  echo "       --info=Information file"
  echo "       --tensorflow_path=TensorFlow path (Use externals/tensorflow by default)"
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

TF_DIR="${SCRIPT_PATH}/../../externals/tensorflow"

for i in "$@"
do
  case $i in
    --info=*)
      INFO=${i#*=}
      ;;
    --tensorflow_path=*)
      TF_DIR=${i#*=}
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
  shift
done

if [ -z "$INFO" ]; then
  echo "INFO is unset or set to the empty string"
  usage
  exit 1
fi
if [ -z "$TF_DIR" ]; then
  echo "tensorflow_path is unset or set to the empty string"
  usage
  exit 1
fi

if [ ! -x "$(command -v bazel)" ]; then
  echo "Cannot find bazel. Please install bazel."
  exit 1
fi

source $INFO

if [ -z "$GRAPHDEF_PATH" ]; then
  echo "GRAPHDEF_PATH is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi
if [ -z "$OPTIMIZE_PATH" ]; then
  echo "OPTIMIZE_PATH is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi
if [ -z "$INPUT" ]; then
  echo "INPUT is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi
if [ -z "$OUTPUT" ]; then
  echo "OUTPUT is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi

CUR_DIR=$(pwd)
{
  echo "Enter $TF_DIR"
  pushd $TF_DIR > /dev/null

  bazel run tensorflow/python/tools:optimize_for_inference -- \
  --input="$GRAPHDEF_PATH" \
  --output="$OPTIMIZE_PATH" \
  --frozen_graph=True \
  --input_names="$INPUT" \
  --output_names="$OUTPUT" \
  --toco_compatible=True

  popd

  echo "OUTPUT FILE : $OPTIMIZE_PATH"
}
