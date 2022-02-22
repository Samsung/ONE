import onnx
import sys

if __name__ == '__main__':
    model = onnx.load(sys.argv[1])
    print(model)
