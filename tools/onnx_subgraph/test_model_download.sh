#!/bin/bash

mkdir -p models

cd ./models
wget https://media.githubusercontent.com/media/onnx/models/refs/heads/main/Computer_Vision/resnext26ts_Opset16_timm/resnext26ts_Opset16.onnx --no-check-certificate

onnxsim resnext26ts_Opset16.onnx ../resnet-test.onnx
