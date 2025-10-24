#!/bin/bash

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT_PATH=$SCRIPT_PATH/../..
FLATC=$ROOT_PATH/Product/out/bin/flatc

if [ ! -e "$1" ]; then
  echo "file not exists: $1"
  exit 1
fi

TFLITE_FILE=$1
TFLITE_FILENAME=${TFLITE_FILE##*\/}
TFLITE_JSON=${TFLITE_FILENAME%\.tflite}.json

$FLATC --json --strict-json $ROOT_PATH/externals/tensorflow/tensorflow/lite/schema/schema.fbs -- $TFLITE_FILE
$SCRIPT_PATH/extract.py $TFLITE_JSON
