/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// from tensorflow/compiler/mlir/lite/ir/tfl_op_interfaces.td

#ifndef CIRCLE_OP_INTERFACES
#define CIRCLE_OP_INTERFACES

include "mlir/IR/OpBase.td"

def CIR_Dialect : Dialect {
  let name = "Circle";

  let description = [{
    The Circle dialect.

    This dialect maps to Circle operations.

    Invariants:

    * All values are of Tensor type (in particular, scalars are
      represented using zero-dimensional tensors);
  }];

  let cppNamespace = "::mlir::Circle";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    ParseResult parseOneResultSameOperandTypeOp(OpAsmParser &parser,
                                                OperationState &result);

    void printOneResultOp(Operation *op, OpAsmPrinter &p);

    // Registered hook to materialize a constant operation from a given
    // attribute value with the desired resultant type.
    Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) override;
  }];
}

//===----------------------------------------------------------------------===//
// CIR op interface for sparse operands.

def CIR_SparseOp : OpInterface<"SparseOpInterface"> {
  let description = [{
    Interface for ops that support sparse computation.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the indices of sparse operands.}],
      "std::vector<int>", "GetSparseOperands", (ins)
    >,
    InterfaceMethod<
      [{Returns the supported block size of float sparse operands.}],
      "std::vector<std::vector<int>>", "GetFloatBlockSize", (ins)
    >,
    InterfaceMethod<
      [{Returns the supported block size of quantized sparse operands.}],
      "std::vector<std::vector<int>>", "GetQuantizedBlockSize", (ins)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// CIR runtime type verification of operand/result types.

def CIR_RuntimeVerification : OpInterface<"CirRuntimeVerifyOpInterface"> {
  let description = [{
    Interface to verify Circle runtime op verification.

    This verifies that the converted Circle ops has operand/result type
    supported by the Circle runtime.
  }];

  let methods = [
    StaticInterfaceMethod<
      [{Returns whether the op's operands/results are supported by runtime.}],
      "LogicalResult", "VerifyCirRuntimeConstraints",
      (ins "Operation*":$op, "bool":$emit_error_on_verify_fail)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// CIR arithmetic count interface.

def CIR_ArithmeticCount : OpInterface<"CirArithmeticCountOpInterface"> {
  let description = [{
    Interface for Circle ops to calculate arithmetic count (Multiply-Add Count).
  }];

  let methods = [
    StaticInterfaceMethod<
      [{Returns an integer representing the op's arithmetic count (Multiply-Add
      Count), return -1 if the arithmetic count cannot be determined.}],
       "int64_t", "GetArithmeticCount", (ins "Operation*":$op)
    >,
  ];
}

#endif // CIRCLE_OP_INTERFACES
