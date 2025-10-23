#!/bin/bash

usage()
{
  echo "usage : $0"
  echo "       --info=Information file"
  echo "       [--tensorflow_path=TensorFlow path] (If omitted, the module installed in system will be used by default.)"
}

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

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

if [ ! -x "$(command -v bazel)" ]; then
  echo "Cannot find bazel. Please install bazel."
  exit 1
fi

source $INFO

if [ -z "$SAVED_MODEL" ] && [ -z "$META_GRAPH" ]; then
  echo "SAVED_MODEL or META_GRAPH + CKPT_PATH is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi
if [ ! -z "$META_GRAPH" ] && [ -z "$CKPT_PATH" ]; then
  echo "META_GRAPH is always used with CKPT_PATH"
  echo "CKPT_PATH is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi
if [ -z "$FROZEN_PATH" ]; then
  echo "FROZEN_PATH is unset or set to the empty string"
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
  if [ -e "$TF_DIR" ]; then
    echo "Enter $TF_DIR"
    pushd $TF_DIR > /dev/null
    FREEZE_GRAPH="bazel run tensorflow/python/tools:freeze_graph -- "
  else
    FREEZE_GRAPH="python -m tensorflow.python.tools.freeze_graph "
  fi

  if [ ! -z $SAVED_MODEL ]; then
    $FREEZE_GRAPH \
    --input_saved_model_dir="$SAVED_MODEL" \
    --input_binary=True \
    --output_node_names="$OUTPUT" \
    --output_graph="$FROZEN_PATH"
  else
    $FREEZE_GRAPH \
    --input_meta_graph="$META_GRAPH" \
    --input_checkpoint="$CKPT_PATH" \
    --input_binary=True \
    --output_node_names="$OUTPUT" \
    --output_graph="$FROZEN_PATH"
  fi

  if [ -e "$TF_DIR" ]; then
    popd
  fi

  echo "OUTPUT FILE : $FROZEN_PATH"
}
