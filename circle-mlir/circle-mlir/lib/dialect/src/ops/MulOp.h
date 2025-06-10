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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_MUL_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_MUL_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

// Return true if the given Mul operation has the CPU kernel supported shapes.
bool VerifyMulOpShapeConstraints(MulOp op)
{
  auto element_type = getElementTypeOrSelf(op.getOutput().getType());

  // Allows QI8 and QUI8 inputs up to five dimension broadcasting unless the
  // output type is not QI16. If the output type is Q16, allows only the same
  // shape operands.
  // TODO support Quantized Type

  // Allows I32, I64 and F32 outputs when the operands have valid shapes,
  // which are broadcastable shapes up to four dimension or have same shapes.
  // Allow I1 as BOOL is used converted from ExpandOnnxOp
  if (IsI32Type(element_type) || IsI64Type(element_type) || element_type.isInteger(1) ||
      /*IsQI16Type(element_type) || element_type.isa<ComplexType>() ||*/
      element_type.isF32())
  {
    return VerifyOperandsHaveSameShapesOrBroadcastableShape(
      /*op=*/op.getOperation(), /*indices=*/ArrayRef<unsigned>{0, 1},
      /*max_bcast_rank=*/4);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  // TODO(b/142478136): Handle fused ops.
  if (getFusedActivationFunction() != "NONE")
    return {};

  // This function is performance critical for op fusion patterns, e.g.
  // FuseBinaryOpToPrecedingAffine and FuseMulOrDivWithConv2dOrDepthwiseConv2d.
  // So a few specializations are provided to evaluate the math operation
  // more efficiently.

  // Specialization for f32 type.
  if (getType().cast<ShapedType>().getElementType().isF32())
  {
    return ConstFoldBinaryOp<FloatAttr, float>(getType(), operands[0], operands[1],
                                               [](float a, float b) { return a * b; });
  }

  // Generic fallback with APFloat
  return ConstFoldBinaryOp(
    getType(), operands, [](APFloat a, APFloat b) { return a * b; },
    [](APInt a, APInt b) { return a * b; });
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_MUL_OP_H__
