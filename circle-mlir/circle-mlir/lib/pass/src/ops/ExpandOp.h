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

#ifndef __CIRCLE_MLIR_PASS_OPS_EXPAND_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_EXPAND_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

namespace mlir
{
namespace Circle
{

class ConvExpand : public mlir::OpConversionPattern<mlir::ONNXExpandOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXExpandOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXExpandOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXExpandOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getInput();
    mlir::Value shape = adaptor.getShape();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvExpandOp intype: " << intype << "\n"; });

    mlir::RankedTensorType shtype = shape.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvExpandOp shtype: " << shtype << "\n"; });

    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvExpandOp outtype: " << outtype << "\n"; });

    rewriter.replaceOpWithNewOp<ExpandOnnxOp>(op, op.getType(), input, shape);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_EXPAND_OP_H__
