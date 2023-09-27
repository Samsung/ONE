from test_utils import TestCase
from test_utils import gen_random_tensor


class DepthwiseConv2D_003_Q8(TestCase):
    def __init__(self):
        self.name = _name_

    def generate(self) -> dict:
        json_content = dict()

        # Generate ifm
        json_content['ifm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        # Generate ker
        json_content['ker'] = gen_random_tensor(
            "uint8",  #dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        # Generate bias
        json_content['bias'] = gen_random_tensor(
            "int32",  #dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0,  # quantized_dimension
            (25))  # value_shape

        # Generate ofm
        json_content['ofm'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0)  # quantized_dimension

        return json_content


_name_ = 'DepthwiseConv2D_003_Q8'

_model_ = DepthwiseConv2D_003_Q8()
