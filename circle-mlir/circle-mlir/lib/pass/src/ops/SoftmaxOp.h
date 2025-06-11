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

#ifndef __CIRCLE_MLIR_PASS_OPS_SOFTMAX_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_SOFTMAX_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

namespace mlir
{
namespace Circle
{

class ConvSoftmax : public mlir::OpConversionPattern<mlir::ONNXSoftmaxOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXSoftmaxOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXSoftmaxOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXSoftmaxOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value input = adaptor.getInput();
    const auto axis = adaptor.getAxis();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType ranked_input_type =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvSoftmax ranked_input_type: " << ranked_input_type << "\n"; });

    // Input must have shape
    const auto last_dim = ranked_input_type.getShape().size() - 1;
    if (axis != -1 && axis != last_dim)
    {
      LLVM_DEBUG({ llvm::dbgs() << "Circle only supports softmax for the last axis.\n"; });
      return mlir::failure();
    }

    mlir::RankedTensorType ranked_output_type =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());

    const auto beta = rewriter.getF32FloatAttr(1.f);
    rewriter.replaceOpWithNewOp<SoftmaxOp>(op, op.getType(), input, beta);

    return mlir::success();
  }
};

class ConvSoftmaxV11 : public mlir::OpConversionPattern<mlir::ONNXSoftmaxV11Op>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXSoftmaxV11Op>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXSoftmaxV11Op::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXSoftmaxV11Op op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value input = adaptor.getInput();
    const auto axis = adaptor.getAxis();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType ranked_input_type =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG(
      { llvm::dbgs() << "ConvSoftmaxV11 ranked_input_type: " << ranked_input_type << "\n"; });

    // Input must have shape
    const auto last_dim = ranked_input_type.getShape().size() - 1;
    if (axis != -1 && axis != last_dim)
    {
      LLVM_DEBUG(
        { llvm::dbgs() << "Circle only supports softmax for the last axis:" << axis << "\n"; });
      return mlir::failure();
    }

    CHECK_VALID_RANK_ATLEAST(ranked_input_type, 2);

    mlir::RankedTensorType ranked_output_type =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    CHECK_VALID_RANK_ATLEAST(ranked_output_type, 2);

    const auto beta = rewriter.getF32FloatAttr(1.f);
    rewriter.replaceOpWithNewOp<SoftmaxOp>(op, op.getType(), input, beta);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_SOFTMAX_OP_H__
