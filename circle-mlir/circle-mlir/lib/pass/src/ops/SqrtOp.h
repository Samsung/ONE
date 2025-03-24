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

#ifndef __CIRCLE_MLIR_PASS_OPS_SQRT_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_SQRT_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

namespace mlir
{
namespace Circle
{

class ConvSqrt : public mlir::OpConversionPattern<mlir::ONNXSqrtOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXSqrtOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXSqrtOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXSqrtOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value input = adaptor.getX();

    // Assume output type is same as input type
    // Input must have shape
    rewriter.replaceOpWithNewOp<SqrtOp>(op, op.getType(), input);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_SQRT_OP_H__
