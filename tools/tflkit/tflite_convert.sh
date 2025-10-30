#!/bin/bash

usage()
{
  echo "usage : $0"
  echo "       --info=<infroamtion file>"
  echo "       [ --tensorflow_path=<path> --tensorflow_version=<version> ] (If omitted, the module installed in system will be used by default.)"
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
    --tensorflow_version=*)
      TF_VERSION=${i#*=}
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

if [ -z "$TF_VERSION" ]; then
  if [ -z "$TF_DIR" ]; then
    TF_VERSION=$(python -c 'import tensorflow as tf; print(tf.__version__)')
    echo "TensorFlow version detected : $TF_VERSION"
  else
    echo "tensorflow_version is unset or set to the empty string"
    usage
    exit 1
  fi
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
if [ -z "$TFLITE_PATH" ]; then
  echo "TFLITE_PATH is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi
if [ -z "$INPUT" ]; then
  echo "INPUT is unset or set to the empty string"
  echo "Update the $INFO file"
  exit 1
fi
if [ -z "$INPUT_SHAPE" ]; then
  echo "INPUT_SHAPE is unset or set to the empty string"
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
    TFLITE_CONVERT="bazel run tensorflow/lite/python:tflite_convert -- "
  else
    TFLITE_CONVERT="python -m tensorflow.lite.python.tflite_convert "
  fi

  NAME_LIST=()
  INPUT_SHAPE_LIST=()
  if [ -z "$NAME" ]; then
    NAME_LIST+=("$TFLITE_PATH")
    INPUT_SHAPE_LIST+=("$INPUT_SHAPE")
  else
    for name in $NAME; do
      NAME_LIST[${#NAME_LIST[*]}]="${TFLITE_PATH/NAME/$name}"
    done
    for shape in $INPUT_SHAPE; do
      INPUT_SHAPE_LIST[${#INPUT_SHAPE_LIST[*]}]="$shape"
    done
    if (( ${#NAME_LIST[*]} != ${#INPUT_SHAPE_LIST[*]} )); then
      echo "The number of NAME and INPUT_SHAPE are different"
      echo "Update the $INFO file"
      exit 1
    fi
  fi

  for (( i=0; i < ${#NAME_LIST[@]}; ++i )); do
    if [ "${TF_VERSION%%.*}" = "2" ]; then
      $TFLITE_CONVERT \
      --output_file="${NAME_LIST[$i]}" \
      --graph_def_file="$GRAPHDEF_PATH" \
      --input_arrays="$INPUT" \
      --input_shapes="${INPUT_SHAPE_LIST[$i]}" \
      --output_arrays="$OUTPUT" \
      --allow_custom_ops=true
    else
      $TFLITE_CONVERT \
      --output_file="${NAME_LIST[$i]}" \
      --graph_def_file="$GRAPHDEF_PATH" \
      --input_arrays="$INPUT" \
      --input_shapes="${INPUT_SHAPE_LIST[$i]}" \
      --output_arrays="$OUTPUT" \
      --allow_custom_ops
    fi
  done

  if [ -e "$TF_DIR" ]; then
    popd
  fi

  for (( i=0; i < ${#NAME_LIST[@]}; ++i )); do
    echo "OUTPUT FILE : ${NAME_LIST[$i]}"
  done
}
