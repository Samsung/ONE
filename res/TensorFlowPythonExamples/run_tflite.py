import numpy as np
import tensorflow as tf

tflite_model = "Part_If_Add_Sub_001.tflite"

interpreter = tf.lite.Interpreter(tflite_model)
interpreter.allocate_tensors()
interpreter.invoke()

# Generate random input data.
num_inputs = len(interpreter.get_input_details())
for i in range(num_inputs):
    input_details = interpreter.get_input_details()[i]
    if input_details["dtype"] == np.float32:
        input_data = np.array(
            np.random.random_sample(input_details["shape"]), input_details["dtype"])
    elif input_details["dtype"] == np.uint8:
        input_data = np.array(
            np.random.randint(0, 256, size=input_details["shape"]),
            input_details["dtype"])
    elif input_details["dtype"] == np.bool_:
        input_data = np.array(
            np.random.choice(a=[True, False], size=input_details["shape"]),
            input_details["dtype"])
    else:
        raise SystemExit("Unsupported input dtype")

    interpreter.set_tensor(input_details["index"], input_data)

interpreter.invoke()

for idx in range(len(interpreter.get_output_details())):
    output_details = interpreter.get_output_details()[idx]
    print("output_details", idx, output_details)
