#!/usr/bin/env python

# PyTorch Example manager

import torch
import importlib
import argparse

parser = argparse.ArgumentParser(description='Process PyTorch python examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

for example in args.examples:
    module = importlib.import_module("examples." + example)
    torch.save(module._model_, example + ".pth")
    print("Generate '" + example + ".pth' - Done")

    torch.onnx.export(module._model_, module._dummy_, example + ".onnx")
    print("Generate '" + example + ".onnx' - Done")
