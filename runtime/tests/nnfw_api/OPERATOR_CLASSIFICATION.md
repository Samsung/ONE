# NNFW API Operator Test Classification Guide

## Overview

This document describes the classification strategy for NNFW API operator tests to improve linking time and maintainability. The classification is based on Circle Schema's BuiltinOperator enum and focuses on functional grouping while excluding unsupported operators.

## Problem Statement

The current NNFW API test implementation has all operator tests, causing excessive linking time if tests are in a single executable file. With 165+ supported operators and growing, this approach doesn't scale well.

## Solution Overview

### Two-Pronged Approach

1. **Solution 1**: Test Group Separation
   - Divide operators into 10 functional groups
   - Create separate executable files for each group
   - Enable selective testing and parallel builds

2. **Solution 2**: Shared Test Utilities Library
   - Extract common test utilities into a shared library
   - Reduce code duplication and compilation time
   - Improve link time by pre-compiling common code

## Operator Classification

### Classification Principles

1. **Functional Similarity**: Operators with similar functionality are grouped together
2. **Build Time Balance**: Each group should have similar compilation complexity
3. **Maintainability**: Related operators should be physically close for easier maintenance
4. **Extensibility**: New operators can be easily categorized using decision tree
5. **Test Strategy**: Enable selective testing of specific functional areas

### Excluded Operators

The following operators are excluded from this classification:
- **STABLEHLO operators**: Not supported by runtime and no plans for future support
- **Deprecated operators**: Operators marked as deprecated in Circle Schema
- **DELEGATE operator**: Not supported by runtime and no plans for future support
- **PLACEHOLDER_FOR_GREATER_OP_CODES operator**: Sudo operator, not used in practice

### Final Operator Groups

#### MathOps
**Description**: Basic mathematical, comparison, and bitwise operations

**Operators**:
```
Arithmetic: ADD, SUB, MUL, DIV, FLOOR_DIV, FLOOR_MOD, NEG, POW, SQUARED_DIFFERENCE, ADD_N, ABS, SIGN, SQUARE
Comparison: EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, MAXIMUM, MINIMUM, ATAN2
Trigonometric: SIN, COS, EXP, LOG, SQRT, RSQRT
Logical: LOGICAL_OR, LOGICAL_AND, LOGICAL_NOT
Bitwise: BITWISE_XOR, RIGHT_SHIFT
Complex: COMPLEX_ABS, IMAG, REAL, RFFT2D
```

**Characteristics**:
- Element-wise operations
- Well-defined mathematical semantics
- Generally simple to test
- Foundation for more complex operations

#### NeuralNetworkOps
**Description**: Core neural network layer operations

**Operators**:
```
Convolution: CONV_2D, DEPTHWISE_CONV_2D, TRANSPOSE_CONV, CONV_3D, CONV_3D_TRANSPOSE
Fully Connected: FULLY_CONNECTED, BCQ_FULLY_CONNECTED
Pooling: AVERAGE_POOL_2D, MAX_POOL_2D, L2_POOL_2D
Matrix: BATCH_MATMUL, MATRIX_DIAG, MATRIX_SET_DIAG, DILATE, REDUCE_WINDOW
Recurrent: LSTM, UNIDIRECTIONAL_SEQUENCE_LSTM, BIDIRECTIONAL_SEQUENCE_LSTM, RNN, UNIDIRECTIONAL_SEQUENCE_RNN, BIDIRECTIONAL_SEQUENCE_RNN, GRU
Sequence: SVDF
Attention: ROPE
Normalization: INSTANCE_NORM
```

**Characteristics**:
- Core building blocks of neural networks
- Often involve weight matrices and complex parameters
- Performance-critical operations
- May have various optimization options

#### ActivationNormOps
**Description**: Activation functions and normalization layers

**Operators**:
```
Activation: RELU, RELU6, RELU_N1_TO_1, LEAKY_RELU, ELU, TANH, LOGISTIC, HARD_SWISH, GELU, RELU_0_TO_1, PRELU
Normalization: L2_NORMALIZATION, LOCAL_RESPONSE_NORMALIZATION, RMS_NORM
Softmax: SOFTMAX, LOG_SOFTMAX
```

**Characteristics**:
- Non-linear transformations
- Often applied element-wise
- Crucial for neural network training and inference
- Usually have simple implementations but important for model behavior

#### TensorDataOps
**Description**: Operations that transform tensor data structures, shapes, layouts, and manipulate tensor data

**Operators**:
```
Shape Operations: RESHAPE, SQUEEZE, EXPAND_DIMS, SHAPE, RANK, ZEROS_LIKE, FILL, BROADCAST_TO
Spatial Transform: DEPTH_TO_SPACE, SPACE_TO_DEPTH, BATCH_TO_SPACE_ND, SPACE_TO_BATCH_ND, BROADCAST_ARGS, RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR
Permutation: TRANSPOSE, TILE
Concat/Split: CONCATENATION, SPLIT, SPLIT_V, CONCAT_EMBEDDINGS
Segment: SEGMENT_SUM, UNIQUE
Pack/Unpack: PACK, UNPACK
Data Generation: ONE_HOT, BUCKETIZE
Padding: PAD, PADV2, MIRROR_PAD
Data Manipulation: GATHER_ND, SCATTER_ND, SPARSE_TO_DENSE, DENSIFY, WHERE
Reverse/Manipulation: REVERSE_V2, REVERSE_SEQUENCE, STRIDED_SLICE, SLICE
Advanced: DYNAMIC_UPDATE_SLICE
```

**Characteristics**:
- Change tensor shape, layout, and data structure without changing core data semantics
- Combine, split, and manipulate multiple tensors
- Often used for data preprocessing and model architecture adjustments
- May involve complex index calculations and boundary conditions
- Important for memory layout optimizations and flexible model architectures

#### IndexingDataOps
**Description**: Data indexing, lookup, generation, and selection operations

**Operators**:
```
Gather: GATHER, GATHER_ND, EMBEDDING_LOOKUP, EMBEDDING_LOOKUP_SPARSE, BCQ_GATHER
Hash: HASHTABLE_LOOKUP, HASHTABLE_FIND, HASHTABLE_IMPORT, HASHTABLE_SIZE, HASHTABLE
Generation: RANGE, FILL, ZEROS_LIKE, RANDOM_UNIFORM, RANDOM_STANDARD_NORMAL, MULTINOMIAL
Selection: SELECT, SELECT_V2, WHERE
```

**Characteristics**:
- Retrieve data based on indices or keys
- Often used in embedding layers and data retrieval
- Generate data patterns and random values
- May involve complex data structures
- Important for efficient data access patterns and dynamic data generation

#### ReductionStatsOps
**Description**: Reduction operations and statistical computations

**Operators**:
```
Reduction: MEAN, SUM, REDUCE_PROD, REDUCE_MAX, REDUCE_MIN, REDUCE_ANY, REDUCE_ALL
Cumulative: CUMSUM
Segment: SEGMENT_SUM, UNSORTED_SEGMENT_SUM, UNSORTED_SEGMENT_PROD, UNSORTED_SEGMENT_MIN, UNSORTED_SEGMENT_MAX
Extrema: ARG_MAX, ARG_MIN, TOPK_V2
Info: RANK, UNIQUE
```

**Characteristics**:
- Reduce tensor dimensions by aggregation
- Compute statistical properties of tensor data
- Often used in model heads and loss computations
- May involve complex reduction patterns along specific axes

#### ControlFlowOps
**Description**: Control flow and conditional execution operations

**Operators**:
```
Conditional: IF, SELECT, SELECT_V2, WHERE
Loop: WHILE
Function: CALL, CALL_ONCE, RUN_MODEL
```

**Characteristics**:
- Control execution flow within models
- Enable conditional and iterative computation
- Complex test scenarios due to multiple execution paths
- Important for dynamic and flexible model architectures

#### TypeConversionOps
**Description**: Data type conversion and quantization operations

**Operators**:
```
Type Cast: CAST, BITCAST
Quantization: DEQUANTIZE, QUANTIZE, FAKE_QUANT
Rounding: CEIL, ROUND, FLOOR
```

**Characteristics**:
- Convert between different data types
- Handle quantization for model compression
- Preserve or change data semantics during conversion
- Critical for model optimization and deployment

#### AdvancedSpecialOps
**Description**: Advanced and special-purpose operations

**Operators**:
```
Detection: NON_MAX_SUPPRESSION_V4, NON_MAX_SUPPRESSION_V5, DETECTION_POSTPROCESS
Variable: VAR_HANDLE, READ_VARIABLE, ASSIGN_VARIABLE
Custom: CUSTOM, LSH_PROJECTION, SKIP_GRAM
```

**Characteristics**:
- Specialized operations for specific use cases
- Often complex with many parameters
- May involve external system interactions
- Important for advanced model capabilities

## Operator Addition Decision Tree

When adding a new operator test, use this decision tree to determine the appropriate group:

```
New Operator
├── Is it a basic math operation?
│   ├── Arithmetic/Comparison/Logical → MathOps
│   └── Bitwise operation → MathOps
├── Is it a core neural network layer?
│   ├── Convolution/Pooling/Matrix → NeuralNetworkOps
│   ├── Recurrent/Sequence → NeuralNetworkOps
│   └── Attention → NeuralNetworkOps
├── Is it an activation or normalization function?
│   ├── Activation function → ActivationNormOps
│   └── Normalization → ActivationNormOps
├── Does it transform tensor data structure/shape/layout or manipulate tensor data?
│   ├── Shape change → TensorDataOps
│   ├── Spatial transform → TensorDataOps
│   ├── Permutation → TensorDataOps
│   ├── Concat/Split → TensorDataOps
│   ├── Pack/Unpack → TensorDataOps
│   ├── Padding → TensorDataOps
│   ├── Data manipulation → TensorDataOps
│   └── Reverse/Strided slice → TensorDataOps
├── Does it index/lookup/generate/select data?
│   ├── Gather/Embedding → IndexingDataOps
│   ├── Hash table → IndexingDataOps
│   ├── Data generation → IndexingDataOps
│   ├── Random generation → IndexingDataOps
│   └── Data selection → IndexingDataOps
├── Does it reduce dimensions?
│   ├── Reduction → ReductionStatsOps
│   ├── Statistics → ReductionStatsOps
│   └── Extrema → ReductionStatsOps
├── Does it control execution flow?
│   ├── Conditional → ControlFlowOps
│   ├── Loop → ControlFlowOps
│   └── Function call → ControlFlowOps
├── Does it convert data types?
│   ├── Type cast → TypeConversionOps
│   ├── Quantization → TypeConversionOps
│   └── Rounding → TypeConversionOps
└── Is it advanced/special purpose?
    ├── Detection → AdvancedSpecialOps
    ├── Variable → AdvancedSpecialOps
    └── Custom → AdvancedSpecialOps
```

## Implementation Guidelines

### Directory Structure

```
runtime/tests/nnfw_api/src/GenModelTests/
├── one_op_tests/
│   ├── MathOps/
│   ├── NeuralNetworkOps/
│   ├── ActivationNormOps/
│   ├── TensorDataOps/
│   ├── IndexingDataOps/
│   ├── ReductionStatsOps/
│   ├── ControlFlowOps/
│   ├── TypeConversionOps/
│   └── AdvancedSpecialOps/
├── one_op_trains/
├── nontrainable_op_trains/
└── NNPackageTests/
```

### Test File Naming Convention

- Use operator name: `{OPERATOR_NAME}.test.cc`
- For multiple related operators: `{CATEGORY}_{OPERATORS}.test.cc`
- Example: `ADD.test.cc`, `CONVOLUTION_POOLING.test.cc`

### CMake Integration

Each group will have its own executable:
- `nnfw_api_gtest_MathOps`
- `nnfw_api_gtest_NeuralNetworkOps`
- ... etc.

### Migration Strategy

1. **Phase 1**: Implement CMake changes and shared library
2. **Phase 2**: Move existing tests to new structure
3. **Phase 3**: Add new operator tests following the classification
4. **Phase 4**: Update CI/CD pipelines for parallel testing

## Expected Benefits

### Build Time Improvement
- **Current**: Single executable with 165+ operators (10+ minutes linking)
- **After**: 10 smaller executables with 7-31 operators each (1-2 minutes linking)
- **Expected improvement**: 80-90% reduction in linking time

### Development Workflow
- **Selective testing**: Test only specific functional areas
- **Parallel builds**: Build multiple test executables simultaneously
- **Faster feedback**: Reduced build times for development iterations
- **Better maintainability**: Clear separation of concerns

### Testing Strategy
- **Focused testing**: Test specific operator groups during development
- **Integration testing**: Run all groups for full validation
- **Performance testing**: Test specific groups for performance regression
- **CI/CD optimization**: Parallel test execution in pipelines

## Maintenance Guidelines

### Adding New Operators
1. Consult the decision tree for group assignment
2. Create test file in appropriate directory
3. Update CMakeLists.txt if new group needed
4. Update this document if classification changes

### Group Rebalancing
- Monitor build times across groups
- Consider rebalancing if build times become uneven
- Update documentation when groups are modified

### Deprecation
- Mark deprecated operators in documentation
- Consider moving deprecated operators to separate group
- Update classification when operators are removed

## Future Considerations

### Scalability
- Current classification supports 165+ operators
- Can accommodate additional operators through existing groups
- New groups can be added for fundamentally different operation types

### Performance Optimization
- Group-specific compiler optimizations
- Link-time optimization tuning per group
- Memory usage optimization for large test suites

### Tool Integration
- IDE integration for group-based test navigation
- Test runner tools with group selection
- Build system integration for incremental builds

## Conclusion

This operator classification provides a scalable, maintainable, and efficient structure for NNFW API testing. By implementing both test group separation and shared test utilities, we can significantly improve build times while maintaining comprehensive test coverage for all supported operators.

The classification is designed to be intuitive for developers, extensible for future operators, and optimized for both development workflows and CI/CD pipelines.
