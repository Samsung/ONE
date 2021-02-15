# Release Note 1.13.0

## ONE Compiler

### Compiler Frontend

- Add optimization pass: ConvertNCHWToNHWC, FoldSparseToDensePass, FuseBatchNormWithConvPass, ForwardReshapeToUnaryOpPass, RemoveUnnecessarySlicePass, RemoveUnnecessarySplitPass,  RemoveUnnecessaryReshapePass, RemoveRedundantReshape, SubstituteTransposeToReshapePass, SubstituteSqueezeToReshapePass, 
- Support more operators: FAKE_QUANT
- Enhancements: Support auto generated random input for record-minmax (for better quantization testing)
- Changes: `--all` option to `--O1` in circle2circle(and one-optimize)
- Fixes: `tf2tfliteV2` accept input shapes `--v2` option, lots of fixes for increase test coverage
- Experimental: Compile ONNX models to circle
