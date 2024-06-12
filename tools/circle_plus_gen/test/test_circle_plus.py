import unittest
import hashlib
import os

from lib.circle_plus import CirclePlus


def get_md5sum(file: str):
    with open(file, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


class TestCirclePlus(unittest.TestCase):
    def setUp(self):
        # sample.circle has a single operator(fully connected)
        circle_file = "./example/sample.circle"
        assert get_md5sum(circle_file) == 'df287dea52cf5bf16bc9dc720e8bca04'

        self.circle_model = CirclePlus.from_file(circle_file)

    def test_get_number_of_operators(self):
        num_op = self.circle_model.get_number_of_operators()
        self.assertEqual(num_op, 1)

    def test_get_operator_names(self):
        ops = self.circle_model.get_operator_names()
        self.assertEqual(ops, ['FULLY_CONNECTED'])


if __name__ == '__main__':
    unittest.main()
