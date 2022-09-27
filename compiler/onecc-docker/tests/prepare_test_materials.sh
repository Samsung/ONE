#!/bin/bash

# Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

# See https://github.com/Samsung/ONE/issues/4155 for information

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd $SCRIPT_PATH > /dev/null

if [[ ! -s "mobilenet_v1_0.5_224.tflite" ]]; then
    rm -rf inception_v3_2018_04_27.tgz
    wget wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_224.tgz
    tar zxvf mobilenet_v1_0.5_224.tgz
fi

popd > /dev/null
