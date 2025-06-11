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

#ifndef __CIRCLE_MLIR_PASS_OPS_SHAPE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_SHAPE_OP_H__

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

class ConvShape : public mlir::OpConversionPattern<mlir::ONNXShapeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXShapeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXShapeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXShapeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();

    LLVM_DEBUG({
      int64_t start = adaptor.getStart();
      llvm::dbgs() << "ConvShape start: " << start << "\n";
    });
    LLVM_DEBUG({
      std::optional<int64_t> end = adaptor.getEnd();
      if (end.has_value())
        llvm::dbgs() << "ConvShape end: " << end.value() << "\n";
    });

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());

    rewriter.replaceOpWithNewOp<ShapeOp>(op, op.getType(), input);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_SHAPE_OP_H__
