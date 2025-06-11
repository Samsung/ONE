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

#ifndef __CIRCLE_MLIR_PASS_OPS_CONCAT_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_CONCAT_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvConcat : public mlir::OpConversionPattern<mlir::ONNXConcatOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXConcatOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXConcatOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXConcatOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::ValueRange inputs = adaptor.getInputs();
    int64_t axis = adaptor.getAxis();

    for (size_t i = 0; i < inputs.size(); ++i)
    {
      mlir::Value input = inputs[i];
      mlir::RankedTensorType intype =
        mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
      LLVM_DEBUG({ llvm::dbgs() << "ConvConcat [" << i << "] intype: " << intype << "\n"; });
    }

    if (inputs.size() == 1)
    {
      rewriter.replaceOp(op, adaptor.getInputs()[0]);
      return mlir::success();
    }

    // Assume output type is same as input type
    // Input must have shape
    rewriter.replaceOpWithNewOp<ConcatenationOp>(op, op.getType(), inputs, axis, "NONE");

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_CONCAT_OP_H__
