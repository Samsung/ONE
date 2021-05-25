#!/usr/bin/env bash
''''export SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"                  # '''
''''export PY_PATH=${SCRIPT_PATH}/../bin/venv/bin/python                                # '''
''''test -f ${PY_PATH} && exec ${PY_PATH} "$0" "$@"                                     # '''
''''echo "Error: Virtual environment not found. Please run 'one-prepare-venv' command." # '''
''''exit 255                                                                            # '''

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

import sys
import tensorflow as tf

if len(sys.argv) != 2:
    print("ERROR: ./resave_savedmodel.py saved_model_path")
    sys.exit()


file_path = sys.argv[1]

MODEL_DIR = file_path
RESAVE_DIR = file_path + "-resaved"
SIGNATURE_KEYS = ['default']
SIGNATURE_TAGS = set()

saved_model = tf.saved_model.load(MODEL_DIR, tags=SIGNATURE_TAGS)

tf.saved_model.save(saved_model, RESAVE_DIR, signatures=saved_model.signatures)

print("Done: " + RESAVE_DIR + " saved")
