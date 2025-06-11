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

#ifndef __CIRCLE_MLIR_PASS_OPS_CLIP_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_CLIP_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/IR/Matchers.h> // from @llvm-project
#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvClip : public mlir::OpConversionPattern<mlir::ONNXClipOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXClipOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXClipOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXClipOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getInput();
    mlir::Value imin = adaptor.getMin();
    mlir::Value imax = adaptor.getMax();

    auto op_name = GetOperationName(op.getOperation());

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    mlir::RankedTensorType mintype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(imin.getType());
    mlir::RankedTensorType maxtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(imax.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvClip name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvClip intype: " << intype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvClip mintype: " << mintype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvClip maxtype: " << maxtype << "\n"; });

    mlir::Location opLoc = op->getLoc();

    // if imin = scalar 0.0 and imax = scalar 6.0 then convert to Relu6
    // else convert to min(max(input,imin), imax)
    // NOTE we can convert to min(max(input, imin), imax) and then
    //      concert to relu6(input) in optimization.

    bool conv2relu6 = false;
    std::vector<float> min_values, max_values;
    if (ExtractConstantValues(imin, min_values) && ExtractConstantValues(imax, max_values))
    {
      if (min_values.size() == 1 && max_values.size() == 1)
      {
        if (min_values[0] == 0.0 && max_values[0] == 6.0)
          conv2relu6 = true;
      }
    }

    if (conv2relu6)
    {
      rewriter.replaceOpWithNewOp<Relu6Op>(op, op.getType(), input);
    }
    else
    {
      mlir::Location min_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/min"));
      auto min_op = rewriter.create<MinimumOp>(min_loc, op.getType(), input, imax);
      rewriter.replaceOpWithNewOp<MaximumOp>(op, op.getType(), min_op, imin);
    }

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_CLIP_OP_H__
