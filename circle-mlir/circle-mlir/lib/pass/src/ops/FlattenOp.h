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

#ifndef __CIRCLE_MLIR_PASS_OPS_FLATTEN_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_FLATTEN_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>
#include <vector>

namespace mlir
{
namespace Circle
{

class ConvFlatten : public mlir::OpConversionPattern<mlir::ONNXFlattenOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXFlattenOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXFlattenOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXFlattenOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getInput();
    int64_t axis = adaptor.getAxis();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();

    auto op_name = GetOperationName(op.getOperation());

    // NOTE use output shape to get shape
    // TODO revise to use axis?
    auto oshape = outtype.getShape();
    mlir::Value shape32 = CreateI32Const(rewriter, oshape, op_name + "/shape");

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), input, shape32);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_FLATTEN_OP_H__
