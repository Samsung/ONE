#!/usr/bin/env python

# PyTorch .pth file dumper

import torch
import argparse

parser = argparse.ArgumentParser(description='Dump pth file')

parser.add_argument('files', metavar='FILES', nargs='+')

args = parser.parse_args()

for file in args.files:
    print(file, "----------------------------------------------------")
    model = torch.load(file)
    model.eval()
    print(model)
