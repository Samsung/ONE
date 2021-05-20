# Release Note 1.16.0

## ONE Compiler

### Compiler Frontend

- Enable `PadV2` in luci-interpreter and quantization
- Provide `circle-tensordump`, `circledump` as a development tool
- Provide `luci-eval-driver` as test tool
- Enable `STRING` type as constant values in CircleConst
- Fix CircleCustom may have 0 input, 0 output
- Enable debian package generation
- More optimization pass
   - Min(6)+ReLU to ReLU6
   - Remove FakeQuant Op
- Experimental support of ONNX upgraded to version 1.8.0 with additional patch
- Fix bugs where one-cmds' config file didn't evaluate boolean properly
