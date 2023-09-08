#!/bin/bash

record-minmax --input_model onnx_conv2d_conv2d_split.opt.qr.circle --output_model onnx_conv2d_conv2d_split.opt.rmm.circle --min_percentile 1.0 --max_percentile 99.0

