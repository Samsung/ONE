from common_place import *
import tensorflow as tf


def run_tflite(model, input_path, output_path=''):
    input = read_input(input_path)

    interpreter = tf.contrib.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    save_result(output_path, output_data)


if __name__ == '__main__':
    args = regular_step()

    run_tflite(args.model[0], args.input, args.output_path)
