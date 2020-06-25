# pota-quantization-value-test

`pota-quantization-value-test` checks whether a Circle model listed in `test.lst` is correctly quantized (`pota` denotes post-training affine). The circle models are generated from the recipes saved in `res/TensorFlowLiteRecipes`.

Write `test.local.lst` for local test.

### Test process

#### Step 1. Fake quantization

Run `circle2circle` with `--quantize_dequantize_weights` option.

Dump the fake-quantized model with `circle-tensordump`.

Compare the dumped model with the expected output in "expected_outputs/<model_name>/\<granularity\>/<quantized_type>/fake_quantization/<tensor_name>.json"

The expected output should include
 (1) values of weights (only for conv, transposed_conv, depthwise_conv, and fc layers)

#### Step 2. Record moving avg of min and moving avg of max for activations

Run `record-minmax` with the fake-quantized model (input data is saved in "test_inputs/<model_name>/\<granularity\>/<quantized_type>/<record_number>.txt")

Dump the minmax-recorded model with `circle-tensordump`.
Compare the dumped model with the expected output in "expected_outputs/<model_name>/\<granularity\>/<quantized_type>/record_minmax/<tensor_name>.json"

The expected output should include
 (1) min/max of activations

#### Step 3. Quantization

Run `circle2circle` with `--quantize_with_minmax` option.

Dump the quantized model with `circle-tensordump`.

Compare the dumped model with the expected output in "expected_outputs/<model_name>/\<granularity\>/<quantized_type>/quantization/<tensor_name>.json"

The expected output should include
 (1) scale, zero point of activations
 (2) scale, zero point, values of weights
 (3) scale, values (weights) of bias
