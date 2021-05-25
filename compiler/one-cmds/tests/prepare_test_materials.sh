#!/bin/bash

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

# See https://github.com/Samsung/ONE/issues/4155 for information

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd $SCRIPT_PATH > /dev/null

if [[ ! -s "inception_v3.pb" ]]; then
    rm -rf inception_v3_2018_04_27.tgz
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
    tar zxvf inception_v3_2018_04_27.tgz
fi

if [[ ! -s "while_3.pbtxt" ]]; then
    rm -rf while_3.zip
    wget https://github.com/Samsung/ONE/files/5095630/while_3.zip
    unzip while_3.zip
    # https://github.com/Samsung/ONE/issues/4155#issuecomment-689320297
fi

if [[ ! -s "mobilenet_test_data.h5" ]]; then
    rm -rf mobilenet_test_data.zip
    wget https://github.com/Samsung/ONE/files/5139460/mobilenet_test_data.zip
    unzip mobilenet_test_data.zip
    # https://github.com/Samsung/ONE/issues/4155#issuecomment-689321538
fi

if [[ ! -s "bcq.pb" ]]; then
    rm -rf bcq.pb.zip
    wget https://github.com/Samsung/ONE/files/5153842/bcq.pb.zip
    unzip bcq.pb.zip
    # https://github.com/Samsung/ONE/issues/4155#issuecomment-689324597
fi

if [[ ! -s "img_files" ]]; then
    rm -rf img_files.zip
    wget https://github.com/Samsung/ONE/files/5499172/img_files.zip
    unzip img_files.zip
    # https://github.com/Samsung/ONE/issues/3213#issuecomment-722757499
fi

if [ ! -d "raw_files" ] || [ ! -s "datalist.txt" ]; then
    ../bin/venv/bin/python preprocess_images.py
fi

if [[ ! -s "inception_v3_test_data.h5" ]]; then
  ../bin/venv/bin/python ../bin/rawdata2hdf5 \
  --data_list datalist.txt \
  --output_path inception_v3_test_data.h5
fi

if [[ ! -d "test_saved_model" ]]; then
    rm -rf test_saved_model.zip
    wget https://github.com/Samsung/ONE/files/5516226/test_saved_model.zip
    unzip test_saved_model.zip
    # https://github.com/Samsung/ONE/issues/4268#issuecomment-724578237
fi

if [[ ! -s "test_keras_model.h5" ]]; then
    rm -rf test_keras_model.zip
    wget https://github.com/Samsung/ONE/files/5520777/test_keras_model.zip
    unzip test_keras_model.zip
    # https://github.com/Samsung/ONE/issues/4268#issuecomment-725025805
fi

if [[ ! -s "test_onnx_model.onnx" ]]; then
    rm -rf test_onnx_model.zip
    wget https://github.com/Samsung/ONE/files/5768243/test_onnx_model.zip
    unzip test_onnx_model.zip
    # https://github.com/Samsung/ONE/issues/5548#issuecomment-754373360
fi

if [[ ! -s "onnx_conv2d_conv2d.onnx" ]]; then
    rm -rf onnx_conv2d_conv2d.zip
    wget https://github.com/Samsung/ONE/files/5774648/onnx_conv2d_conv2d.zip
    unzip onnx_conv2d_conv2d.zip
    # https://github.com/Samsung/ONE/issues/5577#issuecomment-755078444
fi

# public TF models

if [[ ! -s "lite-model_esrgan-tf2_1.tflite" ]]; then
    wget https://tfhub.dev/captain-pool/lite-model/esrgan-tf2/1?lite-format=tflite \
        -O lite-model_esrgan-tf2_1.tflite
    # https://tfhub.dev/captain-pool/lite-model/esrgan-tf2/1
fi

if [[ ! -f "faster_rcnn_openimages_v4_inception_resnet_v2_1/saved_model.pb" ]]; then
    wget https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1?tf-hub-format=compressed \
        -O faster_rcnn_openimages_v4_inception_resnet_v2_1.tar.gz
    mkdir -p faster_rcnn_openimages_v4_inception_resnet_v2_1
    tar zxvf faster_rcnn_openimages_v4_inception_resnet_v2_1.tar.gz \
        -C faster_rcnn_openimages_v4_inception_resnet_v2_1/
    # https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
fi

if [[ ! -f "faster_rcnn_openimages_v4_inception_resnet_v2_1/saved_model.pb" ]]; then
    wget https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2?tf-hub-format=compressed \
        -O efficientnet_lite0_feature-vector_2.tar.gz
    mkdir -p efficientnet_lite0_feature-vector_2
    tar zxvf efficientnet_lite0_feature-vector_2.tar.gz -C efficientnet_lite0_feature-vector_2/
    # https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
fi

if [[ ! -f "experts_bit_r50x1_in21k_object_1/saved_model.pb" ]]; then
    wget https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1?tf-hub-format=compressed \
        -O experts_bit_r50x1_in21k_object_1.tar.gz
    mkdir -p experts_bit_r50x1_in21k_object_1
    tar zxvf experts_bit_r50x1_in21k_object_1.tar.gz -C experts_bit_r50x1_in21k_object_1/
    # ./resave_savedmodel.py experts_bit_r50x1_in21k_object_1
    # https://tfhub.dev/google/experts/bit/r50x1/in21k/object/1
fi

if [[ ! -f "i3d-kinetics-400_1-resaved/saved_model.pb" ]]; then
    wget https://tfhub.dev/deepmind/i3d-kinetics-400/1?tf-hub-format=compressed \
        -O i3d-kinetics-400_1.tar.gz
    mkdir i3d-kinetics-400_1
    tar zxvf i3d-kinetics-400_1.tar.gz -C i3d-kinetics-400_1/
    ./resave_savedmodel.py i3d-kinetics-400_1
    # https://tfhub.dev/deepmind/i3d-kinetics-400/1
fi

if [[ ! -f "tiny_video_net_tvn3_1/saved_model.pb" ]]; then
    wget https://tfhub.dev/google/tiny_video_net/tvn3/1?tf-hub-format=compressed \
        -O tiny_video_net_tvn3_1.tar.gz
    mkdir tiny_video_net_tvn3_1
    tar zxvf tiny_video_net_tvn3_1.tar.gz -C tiny_video_net_tvn3_1/
    # ./resave_savedmodel.py tiny_video_net_tvn3_1
    # https://tfhub.dev/google/tiny_video_net/tvn3/1
fi

if [[ ! -f "movinet_a5_base_kinetics-600_classification_1/saved_model.pb" ]]; then
    wget https://tfhub.dev/tensorflow/movinet/a5/base/kinetics-600/classification/1?tf-hub-format=compressed \
        -O movinet_a5_base_kinetics-600_classification_1.tar.gz
    mkdir movinet_a5_base_kinetics-600_classification_1
    tar zxvf movinet_a5_base_kinetics-600_classification_1.tar.gz -C movinet_a5_base_kinetics-600_classification_1/
    # ./resave_savedmodel.py movinet_a5_base_kinetics-600_classification_1
    # https://tfhub.dev/tensorflow/movinet/a5/base/kinetics-600/classification/1
fi

# prepare 'inception_v3.circle' file used for quantization test
inputfile="./inception_v3.pb"
outputfile="./inception_v3.circle"

if [[ ! -s ${outputfile} ]]; then
  ../bin/one-import-tf \
  --input_path ${inputfile} \
  --output_path ${outputfile} \
  --input_arrays input --input_shapes "1,299,299,3" \
  --output_arrays InceptionV3/Predictions/Reshape_1
fi

popd > /dev/null
