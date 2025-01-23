pip install onnx onnxsim

if [ ! -d "./models/" ];then
  mkdir ./models/
  else
  echo "./models path existing"
fi

cd ./models
wget https://media.githubusercontent.com/media/onnx/models/refs/heads/main/Computer_Vision/resnext26ts_Opset16_timm/resnext26ts_Opset16.onnx --no-check-certificate
#wget https://media.githubusercontent.com/media/onnx/models/refs/heads/main/Natural_Language_Processing/xmod_Opset16_transformers/xmod_Opset16.onnx --no-check-certificate

onnxsim resnext26ts_Opset16.onnx ../resnet-test.onnx
#onnxsim xmod_Opset16.onnx ../xmod-transformer-test.onnx

cd ..
