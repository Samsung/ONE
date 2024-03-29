import tensorflow as tf
import numpy as np

print("tensorflow", tf.__version__)

# NOTE This script is to generate reference I/O data from tflite file
#      for kernels of current(2.12.1) Tensorflow lite does not support,
#      like FullyConneccted with F32 I/O + I4 weights.
# NOTE when we upgrade TF that supportes these kernels, we do not need this
#      script and .ref files in 'res/TensorFlowLiteRecipes/'
#
# run:
#    cd build/debug/compiler/common-artifacts
#    python3 ../../../../compiler/luci-ref-value-py-test/gen_ref_data.py

test_model = "FullyConnected_I4_002"

tflite_model = test_model + ".tflite"
ref_data = test_model + ".ref"
newline = "\n"


def array2string(input, separator=",", idx=0):
    content = ""
    for dim in input:
        if isinstance(dim, np.ndarray):
            content = content + array2string(dim, separator, idx)
            idx = idx + 1
        else:
            if idx > 0:
                content = content + separator
            content = content + str(dim)
            idx = idx + 1
    return content


# Build TFLite interpreter.
interpreter = tf.lite.Interpreter(tflite_model)
interpreter.allocate_tensors()

# Read SignatureDef and get output tensor id orders for remapping
full_signatures = interpreter._get_full_signature_list()
full_signatures_outputs_remap = None
if full_signatures != None:
    signature_serving_default = full_signatures.get('serving_default', None)
    if signature_serving_default != None:
        signature_outputs = signature_serving_default['outputs']

        full_signatures_outputs_remap = []
        for index, (key, value) in enumerate(signature_outputs.items()):
            full_signatures_outputs_remap.append(value)

# Generate random input data.
num_inputs = len(interpreter.get_input_details())
for idx in range(num_inputs):
    input_details = interpreter.get_input_details()[idx]
    if input_details["dtype"] == np.float32:
        input_data = np.array(
            np.random.random_sample(input_details["shape"]), input_details["dtype"])
        input_dtype = "float32"
    else:
        assert False, "Unsupported input dtype"

    interpreter.set_tensor(input_details["index"], input_data)

    content = array2string(input_details["shape"], separator=",") + newline
    content = content + input_dtype + newline
    content = content + array2string(input_data, separator=",") + newline
    with open(ref_data + ".input" + str(idx), 'w') as in_file:
        in_file.write(content)

# Do inference
interpreter.invoke()

inpt_output_details = interpreter.get_output_details()
for idx in range(len(inpt_output_details)):
    output_details = inpt_output_details[idx]
    output_tensor = output_details["index"]
    if full_signatures_outputs_remap != None:
        output_tensor = full_signatures_outputs_remap[idx]

    intp_output_data = interpreter.get_tensor(output_tensor)
    if output_details["dtype"] == np.float32:
        output_dtype = "float32"
    else:
        assert False, "Unsupported data type: " + output_details["dtype"]

    content = array2string(output_details["shape"], separator=",") + newline
    content = content + output_dtype + newline
    content = content + array2string(intp_output_data, separator=",") + newline
    with open(ref_data + ".output" + str(idx), 'w') as out_file:
        out_file.write(content)
