/*
 * Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_MLIR_PASS_OPS_RANGE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_RANGE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

namespace mlir
{
namespace Circle
{

class ConvRange : public mlir::OpConversionPattern<mlir::ONNXRangeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXRangeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXRangeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXRangeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value start = adaptor.getStart();
    mlir::Value limit = adaptor.getLimit();
    mlir::Value delta = adaptor.getDelta();

    rewriter.replaceOpWithNewOp<RangeOp>(op, op.getType(), start, limit, delta);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_RANGE_OP_H__
