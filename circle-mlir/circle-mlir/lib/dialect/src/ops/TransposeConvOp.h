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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_TRANSPOSE_CONV_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_TRANSPOSE_CONV_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// TransposeConvOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult TransposeConvOp::verify()
{
  TransposeConvOp op = *this;
  ShapedType output_type = op.getOutput().getType().cast<ShapedType>();
  ShapedType output_shape_type = op.getOutputShape().getType().cast<ShapedType>();
  if (output_type.hasRank() && output_shape_type.hasStaticShape())
  {
    if (output_type.getRank() != output_shape_type.getDimSize(0))
    {
      return op.emitOpError(llvm::formatv("expect output type has rank = {0}, got output type {1}",
                                          output_shape_type.getDimSize(0), output_type));
    }
  }

  mlir::DenseIntElementsAttr output_shape_elements;
  if (!matchPattern(op.getOutputShape(), m_Constant(&output_shape_elements)))
  {
    return success();
  }

  llvm::SmallVector<int64_t, 4> output_shape;
  output_shape.reserve(output_shape_elements.getNumElements());
  for (auto dim : output_shape_elements.getValues<int>())
  {
    output_shape.push_back(dim);
  }

  auto expected_output_type =
    mlir::Circle::GetTypeFromTensorShape(output_shape, output_type.getElementType());
  if (failed(mlir::verifyCompatibleShape(output_type, expected_output_type)))
  {
    return op.emitOpError(
      llvm::formatv("expect output type {0}, got {1}", expected_output_type, output_type));
  }

  return success();
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_TRANSPOSE_CONV_OP_H__
