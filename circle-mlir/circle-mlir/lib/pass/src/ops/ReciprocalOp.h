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

#ifndef __CIRCLE_MLIR_PASS_OPS_RECIPROCAL_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_RECIPROCAL_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

namespace mlir
{
namespace Circle
{

class ConvReciprocal : public mlir::OpConversionPattern<mlir::ONNXReciprocalOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXReciprocalOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXReciprocalOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXReciprocalOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value input = adaptor.getX();

    auto op_name = GetOperationName(op.getOperation());

    // create scalar 1.0 ConstOp
    auto in_dtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    // TODO support other types
    if (not in_dtype.getElementType().isF32())
      return mlir::failure();
    mlir::Location oneLoc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/reciprocal"));
    mlir::RankedTensorType scalar_type = mlir::RankedTensorType::get({}, in_dtype.getElementType());
    auto one_const =
      rewriter.create<ConstOp>(oneLoc, mlir::DenseElementsAttr::get(scalar_type, {1.0f}));

    rewriter.replaceOpWithNewOp<DivOp>(op, op.getType(), one_const, input, "NONE");

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_RECIPROCAL_OP_H__
