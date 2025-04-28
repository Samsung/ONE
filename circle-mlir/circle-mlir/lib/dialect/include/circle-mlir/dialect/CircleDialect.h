/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.h

#ifndef __CIRCLE_MLIR_DIALECT_CIRCLE_DIALECT_H__
#define __CIRCLE_MLIR_DIALECT_CIRCLE_DIALECT_H__

#include <mlir/Bytecode/BytecodeOpInterface.h>           // from @llvm-project
#include <mlir/Dialect/Traits.h>                         // from @llvm-project
#include <mlir/IR/Dialect.h>                             // from @llvm-project
#include <mlir/IR/DialectImplementation.h>               // from @llvm-project
#include <mlir/Interfaces/DerivedAttributeOpInterface.h> // from @llvm-project
#include <mlir/Interfaces/InferTypeOpInterface.h>        // from @llvm-project
#include <mlir/Interfaces/LoopLikeInterface.h>           // from @llvm-project
#include <mlir/Interfaces/SideEffectInterfaces.h>        // from @llvm-project

#include <circle_schema/schema_generated.h>

#include "mlir/CircleOpsDialect.h.inc"
#include "mlir/CircleOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/CircleOpsAttrdefs.h.inc"

namespace mlir
{
namespace Circle
{

// The Control type is a token-like value that models control dependencies
class ControlType : public Type::TypeBase<ControlType, Type, TypeStorage>
{
public:
  using Base::Base;
  static constexpr StringLiteral name = "cir.control";
};

#include "mlir/CircleOpInterface.h.inc"
#include "mlir/CircleShapeInferenceOpInterfaces.h.inc"

} // namespace Circle
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/CircleOps.h.inc"

// Added here to export method in DynamicShapeUtils.cpp
namespace mlir
{
namespace Circle
{

mlir::RankedTensorType GetTypeFromTensorShape(llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                                              mlir::Attribute encoding = {});

} // end namespace Circle
} // end namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_CIRCLE_DIALECT_H__
