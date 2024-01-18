# MODEL

circledump
```
~/github/YongseopKim/ONE (draft/onert_train_pad ✗) ./build/compiler/circledump/circledump ./op.pad/pad.circle
Dump: ./op.pad/pad.circle

===================================================================
Model version: 0
 # sub graphs: 1

Operator Codes: [order] OpCodeName (OpCode Enum)
[0] PAD (code: 34, dep_code: 34, version: 1)

Buffers: B(index) (length) values, if any
B(0) (0)
B(1) (0)
B(2) (32) 00 00 00 00 00 00 00 00 00 00 00 00 0x1 00 00 00 ...
B(3) (0)
B(4) (16) 0x31 0x2e 0x35 0x2e 0x30 00 00 00 00 00 00 00 00 00 00 00 ...
B(5) (84) 0xc 00 00 00 0x8 00 0xe 00 0x8 00 0x4 00 0x8 00 00 00 ...

-------------------------------------------------------------------
Sub-Graph: #0 main

Operands: T(subgraph index : tensor index) TYPE (shape) (shape_signature) B(buffer index) (variable) OperandName
T(0:0) FLOAT32 (1, 28, 28, 1) (-1, 28, 28, 1) B(1) serving_default_zero_padding2d_input:0

T(0:1) INT32 (4, 2) B(2) sequential/zero_padding2d/Pad/paddings

T(0:2) FLOAT32 (1, 29, 29, 1) (-1, 29, 29, 1) B(3) PartitionedCall:0

Operators: O(subgraph index : operator index) OpCodeName
    Option(values) ... <-- depending on OpCode
    I T(tensor index) OperandName <-- as input
    O T(tensor index) OperandName <-- as output
O(0:0) PAD
    I T(0:0) serving_default_zero_padding2d_input:0
    I T(0:1) sequential/zero_padding2d/Pad/paddings
    O T(0:2) PartitionedCall:0

Inputs/Outputs: I(input)/O(output) T(tensor index) OperandName
I T(0:0) serving_default_zero_padding2d_input:0
O T(0:2) PartitionedCall:0

===================================================================
```

ONERT_LOG_ENABLE
```
~/github/YongseopKim/ONE (draft/onert_train_pad ✗) ONERT_LOG_ENABLE=1 ./Product/out/bin/onert_run --modelfile ./op.pad/pad.circle
Model Filename ./op.pad/pad.circle
...
[  LoweredGraph  ] dump before mandatory passes
[   GraphDumper  ] {
[   GraphDumper  ]     %2 =   @0_Pad(  %0,  %1)
[   GraphDumper  ]   Origin(  %0):    0
[   GraphDumper  ]   Origin(  %1):    1
[   GraphDumper  ]   Origin(  %2):    2
[   GraphDumper  ] }
...
```

# PROGRESS

At the beginning
```
~/github/YongseopKim/ONE (draft/onert_train_pad ✗) ./Product/out/bin/onert_train --modelfile ./op.pad/pad.circle
Model Filename ./op.pad/pad.circle
== training parameter ==
- learning_rate   = 0.001
- batch_size      = 32
- loss_info       = {loss = mean squared error, reduction = sum over batch size}
- optimizer       = sgd
========================
Error during nnfw_session::train_prepare : Padoperation is not trainable yet
```

Tasks
- [x] compute/cker/include/cker/train/operation/Pad.h
- [x] compute/cker/src/train/Pad.test.cc
- [ ] runtime/onert/backend/cpu/ops/PadLayer.h
- [x] runtime/onert/backend/train/ops/PadLayer.cc
- [x] runtime/onert/backend/train/ops/PadLayer.h
- [x] runtime/onert/core/include/ir/train/Operations.Include.h
- [x] runtime/onert/core/include/ir/train/Operations.lst
- [x] runtime/onert/core/include/ir/train/operation/Pad.h
- [x] runtime/onert/core/src/ir/train/operation/Pad.cc
- [x] runtime/onert/backend/train/KernelGenerator.cc
- [x] runtime/onert/backend/train/KernelGenerator.h
- [x] runtime/onert/core/src/compiler/train/TrainableOperationConverter.cc
- [x] runtime/onert/core/src/compiler/train/TrainableOperationConverter.h

Now
```
~/github/YongseopKim/ONE (draft/onert_train_pad ✗) ./Product/out/bin/onert_train --modelfile ./op.pad/pad.circle
Model Filename ./op.pad/pad.circle
== training parameter ==
- learning_rate   = 0.001
- batch_size      = 32
- loss_info       = {loss = mean squared error, reduction = sum over batch size}
- optimizer       = sgd
========================
E: not supported random input and expected generator
```
