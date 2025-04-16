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

#ifndef __CIRCLE_MLIR_PASS_OPS_HARD_SIGMOID_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_HARD_SIGMOID_OP_H__

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

class ConvHardSigmoid : public mlir::OpConversionPattern<mlir::ONNXHardSigmoidOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXHardSigmoidOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXHardSigmoidOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXHardSigmoidOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    // NOTE circle doesn't have Hardsigmoid Op so we decompose this here.
    // - onnx_docs = max(0, min(1, alpha * x + beta))
    // - onnx-tf   = relu(min(1, alpha * x + beta))
    // this will follow onnx-tf way and there is one less Const node
    mlir::Value input = adaptor.getX();
    mlir::APFloat alpha = adaptor.getAlpha();
    mlir::APFloat beta = adaptor.getBeta();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvHardsigmoid name: " << op_name << "\n"; });

    mlir::Value alpha2 = CreateConst(rewriter, alpha.convertToFloat(), op_name + "/alpha");
    if (alpha2.getType().isa<mlir::NoneType>())
      return mlir::failure();
    mlir::Value beta2 = CreateConst(rewriter, beta.convertToFloat(), op_name + "/beta");
    if (beta2.getType().isa<mlir::NoneType>())
      return mlir::failure();

    mlir::Value v_one = CreateConst(rewriter, 1.0f, op_name + "/one");
    if (v_one.getType().isa<mlir::NoneType>())
      return mlir::failure();

    mlir::Value mul_a_x = rewriter.create<MulOp>(opLoc, op.getType(), input, alpha2, "NONE");
    mlir::Value add_mul_b = rewriter.create<AddOp>(opLoc, op.getType(), mul_a_x, beta2, "NONE");
    mlir::Value min_add_one = rewriter.create<MinimumOp>(opLoc, op.getType(), add_mul_b, v_one);

    rewriter.replaceOpWithNewOp<ReluOp>(op, op.getType(), min_add_one);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_HARD_SIGMOID_OP_H__
