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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_ADD_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_ADD_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

// Return true if the given Add operation has the CPU kernel supported shapes.
bool VerifyAddOpShapeConstraints(AddOp op)
{
  auto element_type = getElementTypeOrSelf(op.getOutput().getType());

  // Allows F32 and I32 outputs when the operands have valid shapes,
  // which are broadcastable shapes up to four dimensions or have same shapes.
  // TODO support Quantized Type
  if (element_type.isF32() || IsI32Type(element_type) || IsI64Type(element_type))
  {
    return VerifyOperandsHaveSameShapesOrBroadcastableShape(
      /*op=*/op.getOperation(), /*indices=*/ArrayRef<unsigned>{0, 1},
      /*max_bcast_rank=*/4);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE")
    return {};
  return ConstFoldBinaryOp(
    getType(), operands, [](APFloat a, APFloat b) { return a + b; },
    [](APInt a, APInt b) { return a + b; });
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_ADD_OP_H__
