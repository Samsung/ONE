## onnx2circle-rewrite-test

This test validates whether the "RewriteCirclePass" works as expected.

The test checks:
- Expected operators exist in the output.
- Non-expected operators do not exist.

Procedure are like follows.
- load models used only from 'models/net' folder
- models are loaded from 'test.lst' file
- onnx2circle is executed with `--save_ops` option
- `.circle.ops` text file should be generated for each model
- Circle-MLIR operator type names are listed in the text file
- for each model `__init__.py` file, `check_circle_operators()` should exist
- if operators expection is OK, `check_circle_operators()` should return 0
- if not, `check_circle_operators()` should return non zero value
