 ## Feature Highlights

- **ONE** Compiler
  - Compiler supports more operations
  - New command line interface for user interface consistancy
- **ONE** Runtime
  - Runtime CPU backend supports more operations
  - Runtime CPU backend supports more quant8 operations
  - API changes
  - New optimization
  
## ONE Compiler

### Compiler supports more operations

- MatrixDiag, MatrixSetDiag, ReverseSequence, ReverseV2, SegmentSum, SelectV2, SparseToDense, Where

### New command line interface for user interface consistancy

- one-import: imports conventional model files to circle
   - one-import-tf: imports TensorFlow model to circle
   - one-import-tflite: imports TensorFlow lite model to circle
- one-optimize: circle optimize command
- one-quantize: circle quantize command
   - supports float32 to uint8, layer wise (for Conv series)
- one-pack: package command
- one-prepare-venv: prepares python virtual environment for importing TensorFlow model
- one-codegen: backend(if available) code generator

## ONE Runtime

### Runtime CPU backend supports more operations

- LogSoftmax, SpaceToBatchND

### Runtime CPU backend supports more quant8 operations

- Logistic, Mul, Tanh, SpaceToBatchND, Transpose, Sub, Max, Min, Less, Greater, GreaterEqual, LessEqual, Equal, NotEqual

### API changes

- Introduce basic asynchronous execution API

### New optimization
    
- Remove dynamic tensor overhead from static models
