/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_MLIR_PASS_OPT_CONVERT_RESHAPE_SHAPE_32_H__
#define __CIRCLE_MLIR_PASS_OPT_CONVERT_RESHAPE_SHAPE_32_H__

#include "ConvertHelper.h"

#include <cassert>

namespace mlir
{
namespace Circle
{

// Find INT64 Const shape of Reshape
//    Const(shape/INT64)-Reshape
// Relace Const(shape/INT64) with Const(shape/INT32)
//    Const(shape/INT32)-Reshape
struct ConvertReshapeShape32 : public OpRewritePattern<ReshapeOp>
{
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape_op, PatternRewriter &rewriter) const override
  {
    mlir::Operation *is_const = reshape_op.getOperand(1).getDefiningOp();
    if (!mlir::isa_and_nonnull<ConstOp>(is_const))
      return mlir::failure();

    auto const_op = cast<ConstOp>(is_const);
    auto const_type = mlir::cast<TensorType>(const_op.getType());
    if (const_type.getElementType().isInteger(32))
      return mlir::failure();
    assert(const_type.getElementType().isInteger(64));

    mlir::Value reshape_shape = const_op; // ExtractConstantValues requries mlir::Value
    std::vector<int32_t> values;
    if (!ExtractConstantValues(reshape_shape, values))
      return mlir::failure();

    mlir::Location opLoc = const_op->getLoc();
    mlir::RankedTensorType stype =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(const_op.getType());
    mlir::Type i32 = rewriter.getI32Type();
    mlir::RankedTensorType si32stype = RankedTensorType::get(stype.getShape(), i32);
    mlir::Value shape32 =
      rewriter.create<ConstOp>(opLoc, DenseIntElementsAttr::get(si32stype, values));

    auto &shape_mutable = reshape_op.getShapeMutable();
    shape_mutable.assign(shape32);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPT_CONVERT_RESHAPE_SHAPE_32_H__
