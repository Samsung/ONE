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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_PRELU_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_PRELU_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// PReluOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult PReluOp::verify()
{
  PReluOp op = *this;
  auto input_type = op.getInput().getType().cast<ShapedType>();
  auto alpha_type = op.getAlpha().getType().cast<ShapedType>();
  auto output_type = op.getOutput().getType().cast<ShapedType>();

  if (input_type.hasStaticShape() && alpha_type.hasStaticShape())
  {
    if (input_type.getRank() != alpha_type.getRank() + 1)
    {
      return op.emitOpError("'alpha' should have one less rank than 'input'.");
    }

    // Check if alpha is broadcastable
    for (int i = 0; i < alpha_type.getRank(); i++)
    {
      if (alpha_type.getDimSize(i) != input_type.getDimSize(i + 1) && alpha_type.getDimSize(i) != 1)
      {
        return op.emitOpError(llvm::formatv("'alpha' is not broadcastable at dimension {0}.", i));
      }
    }
  }

  if (input_type.hasStaticShape() && output_type.hasStaticShape())
  {
    if (input_type.getRank() != output_type.getRank())
    {
      return op.emitOpError("'input' and 'output' should have the same rank.");
    }

    // Check if input and output shapes are same
    for (int i = 0; i < input_type.getRank(); i++)
    {
      if (input_type.getDimSize(i) != output_type.getDimSize(i))
      {
        return op.emitOpError("'input' and 'output' should have the same shape.");
      }
    }
  }
  return success();
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_PRELU_OP_H__
