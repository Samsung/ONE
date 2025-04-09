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

#ifndef __CIRCLE_MLIR_PASS_OPS_POW_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_POW_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvPow : public mlir::OpConversionPattern<mlir::ONNXPowOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXPowOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXPowOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXPowOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value X = adaptor.getX();
    mlir::Value Y = adaptor.getY();

    std::vector<float> Y_values;
    if (!ExtractConstantValues(Y, Y_values))
    {
      LLVM_DEBUG({ llvm::dbgs() << "ConvPow failed to extract constant value of Y\n"; });
      return mlir::failure();
    }
    assert(Y_values.size() == 1);

    if (Y_values[0] == 2.0)
    {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), X, X, "NONE");
    }
    else
    {
      rewriter.replaceOpWithNewOp<PowOp>(op, op.getType(), X, Y);
    }

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_POW_OP_H__
