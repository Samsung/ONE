import json
import typing
import numpy as np
import os


def _dump_npy_included_json(output_dir: str, json_content: dict):
    """
    Dump json to output_dir, and save all included npy files
    """
    # Create output_dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # file name for npy data (ex: 0.npy, 1.npy, ...)
    _index = 0
    _index_to_value = dict()

    # Replace npy to the path to the npy file
    for tensor_name, qparam in json_content.items():
        assert type(tensor_name) == str
        assert type(qparam) == dict
        for field, value in qparam.items():
            if isinstance(value, np.ndarray):
                npy_name = str(_index) + '.npy'

                # Save npy file
                np.save(os.path.join(output_dir, npy_name), value)

                # Replace to the path to the npy file
                json_content[tensor_name][field] = npy_name

                # Save the mapping from index to tensor name
                _index_to_value[_index] = tensor_name + "_" + field
                _index += 1

    # Dump json
    with open(os.path.join(output_dir, 'qparam.json'), 'w') as f:
        json.dump(json_content, f, indent=2)


def _str_to_npy_dtype(dtype_str: str):
    if dtype_str == "uint8":
        return np.uint8
    if dtype_str == "int16":
        return np.int16
    if dtype_str == "int32":
        return np.int32
    if dtype_str == "int64":
        return np.int64
    raise SystemExit("Unsupported npy dtype", dtype_str)


def gen_random_tensor(dtype_str: str,
                      scale_shape: typing.Tuple[int],
                      zerop_shape: typing.Tuple[int],
                      quantized_dimension: int,
                      value_shape: typing.Optional[typing.Tuple[int]] = None) -> dict:
    content = dict()
    content['dtype'] = dtype_str
    content['scale'] = np.random.rand(scale_shape).astype(np.float32)
    # Why 256? To ensure the smallest dtype (uint8) range [0, 256)
    content['zerop'] = np.random.randint(256, size=zerop_shape, dtype=np.int64)
    content['quantized_dimension'] = quantized_dimension

    if value_shape != None:
        dtype = _str_to_npy_dtype(dtype_str)
        content['value'] = np.random.randint(256, size=value_shape, dtype=dtype)
    return content


class TestCase:
    def __init__(self):
        pass

    def generate(self) -> dict:
        pass


class TestRunner:
    def __init__(self, output_dir: str):
        self.test_cases = list()
        self.output_dir = output_dir

    def register(self, test_case: TestCase):
        self.test_cases.append(test_case)

    def run(self):
        for test_case in self.test_cases:
            print("Generate test case: " + test_case.name)
            _dump_npy_included_json(self.output_dir + '/' + test_case.name,
                                    test_case.generate())
