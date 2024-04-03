import numpy as np
import subprocess
import os


# read input/output data files model_name.ref.input* and
# model_name.ref.output* and return the contents
def recover_fromfile(path, test_name, suffix):
    # .ref file format
    # 1'st line is shape, i.e. "2,4"
    # 2'nd line is dtype, i.e. "float32"
    # 3'rd line is comma seperated values
    ref_filename = test_name + ".ref." + suffix
    ref_datapath = os.path.join(path, ref_filename)

    num_data = 0
    parse_shape = []
    parse_dtype = []
    parse_value = []

    while True:
        refnum_filepath = ref_datapath + str(num_data)
        if (not os.path.isfile(refnum_filepath)):
            break
        with open(refnum_filepath, "r") as ref_file:
            lines = ref_file.readlines()
            assert len(lines) >= 3, "Invalid file: " + ref_filename + str(num_data)
        print("load reference data from", test_name)
        shape = [int(i) for i in lines[0].split(",")]
        dtype = lines[1].strip("\r\n \t")
        if dtype == "float32":
            value = [float(i) for i in lines[2].split(",")]
        else:
            assert False, "Unsupported data type: " + dtype

        # validate shape and number of elements
        num_elements = 1
        for dim in shape:
            num_elements = num_elements * dim
        if num_elements != len(value):
            assert False, "Number of value elements do not match with shape"

        parse_shape.append(shape)
        parse_dtype.append(dtype)
        parse_value.append(value)

        num_data = num_data + 1

    return num_data, parse_shape, parse_dtype, parse_value


def recover_inputs(path, test_name):
    return recover_fromfile(path, test_name, "input")


def recover_outputs(path, test_name):
    return recover_fromfile(path, test_name, "output")


# save reference data to input files for luci-eval-driver
def save_binary_inputs(path, test_name, num_inputs, input_shape, input_dtype, input_data):
    circle_inputpath = os.path.join(path, test_name + ".circle.input")
    for index in range(0, num_inputs):
        # reference input value
        if input_dtype[index] == "float32":
            nps = np.asarray(input_data[index], dtype=np.float32)
            nps.tofile(circle_inputpath + str(index))
        else:
            assert False, "Unsupported data type: " + input_dtype[index]
        # reference input shape
        nps = np.asarray(input_shape[index], dtype=np.short)
        nps.tofile(circle_inputpath + str(index) + ".shape", sep=",")
        # reference input dtype
        with open(circle_inputpath + str(index) + ".dtype", "w") as dtype_file:
            dtype_file.write(input_dtype[index])


def luci_eval_verify(test_name, binary_path, eval_driver, rtolf32=1e-5, atolf32=1e-5):
    circle_model = os.path.join(binary_path, test_name + ".circle")

    num_inputs, input_shape, input_dtype, input_data = recover_inputs(
        binary_path, test_name)
    assert num_inputs > 0, "No valid reference input file"
    save_binary_inputs(binary_path, test_name, num_inputs, input_shape, input_dtype,
                       input_data)

    num_ouputs, output_shape, output_dtype, output_data = recover_outputs(
        binary_path, test_name)
    assert num_ouputs > 0, "No valid reference output file"

    # Execute luci interpreter.
    subprocess.run(
        [
            eval_driver, circle_model,
            str(num_inputs), circle_model + ".input", circle_model + ".output"
        ],
        check=True)

    # Compare the results.
    for idx in range(num_ouputs):
        luci_output_data = np.fromfile(circle_model + ".output" + str(idx),
                                       output_dtype[idx])
        luci_output_data = np.reshape(luci_output_data, output_shape[idx])
        ref_output_data = np.reshape(output_data[idx], output_shape[idx])

        show_vals_and_stop = False
        if output_dtype[idx] == "float32":
            if not np.allclose(
                    luci_output_data, ref_output_data, rtol=rtolf32, atol=atolf32):
                show_vals_and_stop = True
        else:
            assert False, "Unsupported data type: " + output_dtype[idx]

        if show_vals_and_stop:
            print("\nreference:\n", ref_output_data)
            print("luci:\n", luci_output_data)
            message = "Execution result of " + test_name + " does not match with reference"
            assert False, message


# arguments must be in sync with `conftest.py`
def test_luci_eval(default_test_name: str, binary_path: str, eval_driver_path: str):
    luci_eval_verify(default_test_name, binary_path, eval_driver_path)


# arguments must be in sync with `conftest.py`
def test_luci_eval_tol(tol_test_name: str, binary_path: str, eval_driver_path: str,
                       rtolf32: str, atolf32: str):
    luci_eval_verify(tol_test_name, binary_path, eval_driver_path, float(rtolf32),
                     float(atolf32))
