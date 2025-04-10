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

#ifndef __CIRCLE_MLIR_PASS_OPS_MAX_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_MAX_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvMax : public mlir::OpConversionPattern<mlir::ONNXMaxOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXMaxOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXMaxOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXMaxOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    LLVM_DEBUG({ llvm::dbgs() << "ConvMax data_0.size: " << op.getData_0().size() << "\n"; });
    if (op.getData_0().size() != 2)
    {
      // TODO support more than 2 inputs
      return mlir::failure();
    }

    mlir::ValueRange data_0 = op.getData_0();
    mlir::Value input = data_0[0];
    mlir::Value B = data_0[1];

    rewriter.replaceOpWithNewOp<MaximumOp>(op, op.getType(), input, B);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_MAX_OP_H__
