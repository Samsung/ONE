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

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.cc

#ifndef __CIRCLE_MLIR_DIALECT_OPS_CAST_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_CAST_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  assert(operands.size() == 1);
  if (getElementTypeOrSelf(getInput()) == getElementTypeOrSelf(getType()))
  {
    return getInput();
  }

  // For now, only supports cast for the integer/float input type.
  auto elements_attr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(operands[0]);
  if (!elements_attr)
  {
    return nullptr;
  }

  auto result_element_type = mlir::cast<ShapedType>(getType()).getElementType();
  auto operand_element_type = mlir::cast<ShapedType>(getInput().getType()).getElementType();
  auto operand_int_type = mlir::dyn_cast<IntegerType>(operand_element_type);
  if (!result_element_type || !operand_element_type)
  {
    return nullptr;
  }

  if (mlir::isa<mlir::IntegerType>(result_element_type))
  {
    auto result_int_type = mlir::dyn_cast<IntegerType>(result_element_type);
    const int output_bitwidth = result_int_type.getWidth();
    // check for INT64 <--> INT32
    if (operand_int_type)
    {
      const bool is_unsigned = operand_int_type.isUnsigned();
      const bool involves_bool =
        operand_int_type.getWidth() == 1 || result_int_type.getWidth() == 1;
      // The integer cast op is the same as C integer cast. Depends on the operand
      // type's signedness, we will determine whether or not sign extension is
      // needed.
      auto cast = [&](APInt value) {
        if (involves_bool)
        {
          // Handle boolean inputs or outputs explicitly as it doesn't have the same
          // behavior as extension or truncation.
          // true input should always be cast to 1 and not -1 as the sign extension
          // would do for signed outputs. Similarly, non-zero inputs should be cast
          // to true. Truncating even numbers to one bit will result in `false`.
          return APInt(result_int_type.getWidth(), value != 0);
        }
        return is_unsigned ? value.zextOrTrunc(output_bitwidth)
                           : value.sextOrTrunc(output_bitwidth);
      };
      return elements_attr.mapValues(result_int_type, cast);
    }
    // for Float32 --> INT64/INT32
    const bool is_unsigned = result_int_type.isUnsigned();
    auto cast = [&](APFloat value) {
      // reference from llvm-project/mlir/lib/Dialect/Arith/IR/ArithOps.cpp
      bool ignored;
      mlir::APSInt api(output_bitwidth, is_unsigned);
      value.convertToInteger(api, mlir::APFloat::rmTowardZero, &ignored);
      return api;
    };
    return elements_attr.mapValues(result_int_type, cast);
  }
  else if (mlir::isa<mlir::FloatType>(result_element_type))
  {
    // Refer to https://llvm.org/doxygen/classllvm_1_1APFloat.html
    auto result_float_type = mlir::dyn_cast<FloatType>(result_element_type);
    // To get the correct semantics of floating point from the type of this CastOp
    const llvm::fltSemantics &semantics = result_float_type.getFloatSemantics();
    auto cast = [&](const llvm::APInt &value) {
      llvm::APFloat float_value(static_cast<double>(value.getSExtValue()));
      bool loses_info_unused;
      // rmNearestTiesToEven: default rounding mode
      float_value.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &loses_info_unused);
      return float_value.bitcastToAPInt();
    };
    return elements_attr.mapValues(result_float_type, cast);
  }

  return nullptr;
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_CAST_OP_H__
