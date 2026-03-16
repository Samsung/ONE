# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved

import unittest
import os
import sys
import extract_onnx_lib
import shutil


def onnx_parser_test(args):
    #exe = './onnx-subgraph ' + '--onnx=test.onnx'
    exe = './onnx-subgraph ' + args
    rec = os.system(exe)


class ONNX_Parser_Test(unittest.TestCase):
    def test_parse_result_exception(self):
        ret = os.path.exists('./subgraphs_ios.txt')
        if ret:
            os.remove('./subgraphs_ios.txt')
        onnx_parser_test('--onnx=no_file.onnx')
        ret = os.path.exists('./subgraphs_ios.txt')
        self.assertEqual(ret, False)

    def test_parse_result_normal(self):
        ret = os.path.exists('./subgraphs_ios.txt')
        if ret:
            os.remove('./subgraphs_ios.txt')

        onnx_parser_test('--onnx=test.onnx')
        ret = os.path.exists('./subgraphs_ios.txt')
        self.assertEqual(ret, True)

    def test_subgraph_normal(self):
        ret = os.path.exists('./subgraphs')
        if ret:
            shutil.rmtree(path='./subgraphs')

        extract_onnx_lib.split_onnx_ios('./subgraphs_ios.txt', './test.onnx')
        ret = os.path.exists('./subgraphs')
        self.assertEqual(ret, True)

        ret = os.path.exists('./subgraphs/CPU')
        self.assertEqual(ret, True)

        ret = os.path.exists('./subgraphs/NPU')
        self.assertEqual(ret, True)

        ret = os.path.exists('./subgraphs/CPU/CPUsubgraph15.onnx')
        self.assertEqual(ret, True)

        ret = os.path.exists('./subgraphs/NPU/NPUsubgraph15.onnx')
        self.assertEqual(ret, True)

    def test_subgraph_exception(self):
        ret = os.path.exists('./subgraphs')
        if ret:
            shutil.rmtree(path='./subgraphs')

        extract_onnx_lib.split_onnx_ios('./subgraphs_ios.txt', './fake.onnx')
        ret = os.path.exists('./subgraphs')
        self.assertEqual(ret, False)


if __name__ == '__main__':
    unittest.main()
