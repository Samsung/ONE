from common_place import *

import onnx
import caffe2.python.onnx.backend


def run_onnx(model, input_path, output_path=''):  #args.model[0] , args.input
    path = model

    #I'll leave it in case anyone needs to read the .pb file.
    #proto_arr = onnx.TensorProto()
    #with open(input_path, 'rb') as f:
    #    proto_arr.ParseFromString(f.read())
    #    input_arr = onnx.numpy_helper.to_array(proto_arr)

    modelFile = onnx.load(path, 'rb')
    input_arr = read_input(input_path)
    output = caffe2.python.onnx.backend.run_model(modelFile, input_arr)

    print(output)
    save_result(output_path, output)


if __name__ == '__main__':
    args = regular_step()

    run_onnx(args.model[0], args.input, args.output_path)
