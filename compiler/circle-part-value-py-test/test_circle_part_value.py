import numpy as np
import tensorflow as tf
import subprocess
import os
import json


# Compares the execution result of TFLite interpreter and partitioned model(s) from a circle model.
def part_eval(test_name, bin_dir, circle_part_driver):
    artifacts_dir = os.path.join(bin_dir, test_name)
    tflite_model = os.path.join(artifacts_dir, test_name + ".tflite")
    circle_model = os.path.join(artifacts_dir, test_name + ".circle")
    partition_conn_ini = os.path.join(artifacts_dir, test_name + ".conn.ini")
    partition_conn_json = os.path.join(artifacts_dir, test_name + ".conn.json")
    expected_count = os.path.join(artifacts_dir, test_name + ".excnt")

    # Check expected count of models from partitioning
    try:
        with open(expected_count, "r") as expected_count_file:
            expected_count_line = expected_count_file.readline()

        expected_count_line = int(expected_count_line)
        if expected_count_line:
            with open(partition_conn_json) as json_file:
                json_data = json.load(json_file)
                parts_value = json_data["parts"]
                if len(parts_value) != expected_count_line:
                    print("Partitioned model count differs from expected:",
                          expected_count_line)
                    assert False

                print("Partitioned model count expected: ", expected_count_line)
        else:
            print("Skip expected partitioned model count check: 0")

    except:
        print("Skip expected partitioned model count check: error")

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
    for i in range(num_inputs):
        input_details = interpreter.get_input_details()[i]
        if input_details["dtype"] == np.float32:
            input_data = np.array(
                np.random.random_sample(input_details["shape"]), input_details["dtype"])
        elif input_details["dtype"] == np.uint8:
            input_data = np.array(
                np.random.randint(0, 256, size=input_details["shape"]),
                input_details["dtype"])
        elif input_details["dtype"] == np.int16:
            input_data = np.array(
                np.random.randint(0, 100, size=input_details["shape"]),
                input_details["dtype"])
        elif input_details["dtype"] == np.int32:
            input_data = np.array(
                np.random.randint(0, 100, size=input_details["shape"]),
                input_details["dtype"])
        elif input_details["dtype"] == np.int64:
            input_data = np.array(
                np.random.randint(0, 100, size=input_details["shape"]),
                input_details["dtype"])
        elif input_details["dtype"] == np.bool_:
            input_data = np.array(
                np.random.choice(a=[True, False], size=input_details["shape"]),
                input_details["dtype"])
        else:
            assert False, "Unsupported input dtype"

        interpreter.set_tensor(input_details["index"], input_data)
        input_data.tofile(circle_model + ".input" + str(i))

    # Do inference
    interpreter.invoke()

    # Execute circle-part-driver.
    partition_command = [
        circle_part_driver, partition_conn_ini,
        str(num_inputs), circle_model + ".input", circle_model + ".output"
    ]
    print("Run: ")
    for arg in partition_command:
        print("    ", arg, "\\")
    print("", flush=True)

    # working directory into the folder as ini has relative filename of the model
    subprocess.run(partition_command, check=True, cwd=artifacts_dir)

    # Compare the results.
    inpt_output_details = interpreter.get_output_details()
    for idx in range(len(inpt_output_details)):
        output_details = inpt_output_details[idx]
        output_data = np.fromfile(circle_model + ".output" + str(idx),
                                  output_details["dtype"])
        shape_file = open(circle_model + ".output" + str(idx) + ".shape", 'r')
        output_shape = [int(i) for i in shape_file.read().split(',')]
        luci_output_data = np.reshape(output_data, output_shape)
        output_tensor = output_details["index"]
        if full_signatures_outputs_remap != None:
            output_tensor = full_signatures_outputs_remap[idx]
        intp_output_data = interpreter.get_tensor(output_tensor)
        if output_details["dtype"] == np.uint8:
            assert np.allclose(
                luci_output_data, intp_output_data, rtol=0, atol=0
            ), "Execution result of " + tflite_model + " does not match with " + circle_model
        elif output_details["dtype"] == np.float32:
            assert np.allclose(
                luci_output_data, intp_output_data, rtol=1.e-5, atol=1.e-5
            ), "Execution result of " + tflite_model + " does not match with " + circle_model
        elif output_details["dtype"] == np.int64:
            assert np.allclose(
                luci_output_data, intp_output_data, rtol=0, atol=0
            ), "Execution result of " + tflite_model + " does not match with " + circle_model
        elif output_details["dtype"] == np.int32:
            assert np.allclose(
                luci_output_data, intp_output_data, rtol=0, atol=0
            ), "Execution result of " + tflite_model + " does not match with " + circle_model
        elif output_details["dtype"] == np.int16:
            assert np.allclose(
                luci_output_data, intp_output_data, rtol=0, atol=0
            ), "Execution result of " + tflite_model + " does not match with " + circle_model
        elif output_details["dtype"] == np.bool_:
            assert np.allclose(
                luci_output_data, intp_output_data, rtol=0, atol=0
            ), "Execution result of " + tflite_model + " does not match with " + circle_model
        else:
            assert False, "Unsupported data type: " + output_details["dtype"]


# arguments must be in sync with `conftest.py`
def test_circle_part_value(test_name: str, bin_dir: str, part_driver_path: str):
    part_eval(test_name, bin_dir, part_driver_path)
