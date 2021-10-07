# circle-opselector-test

There are two tests

- `arg_validity_test.sh`

  This test checks cli options

- `value_test.sh`

  This test compares the selected circle model by `recipe` and by `circle-opselector`

## value_test process

1. Prepare `tflite recipe` for origin circle model(**origin.circle**). The recipe was already in `res/TensorFlowLiteRecipes` such as Part_Sqrt_Rsqrt_001
2. Select the nodes to be cut from the model(**origin.circle**) and make a `tflite recipe` consisting of only the nodes. Output file is **selected.tflite**
3. Select the node using `circle-opselector`. And output file is **selected.circle**
4. Compare two models(selected.tflite and selected.circle) using `luci-value-test` which compares the out value of models by random input value.

**Improvements(To do)**

- Create more test cases
- Remove the dependecy with `luci-value-test` 
