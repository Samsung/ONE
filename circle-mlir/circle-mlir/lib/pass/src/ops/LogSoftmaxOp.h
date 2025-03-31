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

#ifndef __CIRCLE_MLIR_PASS_OPS_LOG_SOFTMAX_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_LOG_SOFTMAX_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <stdexcept>

namespace mlir
{
namespace Circle
{

// NOTE ONNXLogSoftmaxOp is decomposed to LogOp and SoftmaxOp and thus
//      this conversion is not used.
class ConvLogSoftmax : public mlir::OpConversionPattern<mlir::ONNXLogSoftmaxOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXLogSoftmaxOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXLogSoftmaxOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXLogSoftmaxOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    throw std::runtime_error("NYI ConvLogSoftmax");
    // TODO uncomment when necessary
    /*
    mlir::Value input = adaptor.getInput();
    const auto axis = adaptor.getAxis();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType ranked_input_type =
      input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG(
      { llvm::dbgs() << "ConvLogSoftmax ranked_input_type: " << ranked_input_type << "\n"; });

    // Assume input have shape
    const auto last_dim = ranked_input_type.getShape().size() - 1;
    if (axis != -1 && axis != last_dim)
    {
      LLVM_DEBUG({ llvm::dbgs() << "Circle only supports LogSoftmax for the last axis.\n"; });
      return mlir::failure();
    }

    // Currently only supports rank 2.
    CHECK_VALID_RANK_2(ranked_input_type);

    mlir::RankedTensorType ranked_output_type =
      op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    CHECK_VALID_RANK_2(ranked_output_type);

    // Assume output type is same as input type
    rewriter.replaceOpWithNewOp<LogSoftmaxOp>(op, op.getType(), input);

    return mlir::success();
    */
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_LOG_SOFTMAX_OP_H_
