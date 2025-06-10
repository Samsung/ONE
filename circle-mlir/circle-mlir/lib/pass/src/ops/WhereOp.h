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

#ifndef __CIRCLE_MLIR_PASS_OPS_WHERE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_WHERE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvWhere : public mlir::OpConversionPattern<mlir::ONNXWhereOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXWhereOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXWhereOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXWhereOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getCondition();
    mlir::Value X = adaptor.getX();
    mlir::Value Y = adaptor.getY();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType xtype = X.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType ytype = Y.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();

    if (isSelectOp(intype, xtype, ytype))
    {
      rewriter.replaceOpWithNewOp<SelectOp>(op, op.getType(), input, X, Y);
    }
    else
    {
      rewriter.replaceOpWithNewOp<SelectV2Op>(op, op.getType(), input, X, Y);
    }

    return mlir::success();
  }

private:
  bool isSelectOp(mlir::RankedTensorType &intype, mlir::RankedTensorType &xtype,
                  mlir::RankedTensorType &ytype) const
  {
    // The condition WhereOp can be converted to be Select Op:
    // 1. Either the same shape (in which case the select is elementwise), or
    // 2. condition must be Rank 1 and match over the first dimension.
    // Otherwise, WhereOp is converted SelectV2 Op.
    const auto inshape = intype.getShape();
    const auto xshape = xtype.getShape();
    const auto yshape = ytype.getShape();

    const bool cond_1 = inshape == xshape && inshape == yshape;
    const bool cond_2 = inshape.size() == 1 && xshape.size() >= 1 && yshape.size() >= 1 &&
                        inshape[0] == xshape[0] && inshape[0] == yshape[0];

    return cond_1 || cond_2;
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_WHERE_OP_H__
