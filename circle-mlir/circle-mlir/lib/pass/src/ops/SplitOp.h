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

#ifndef __CIRCLE_MLIR_PASS_OPS_SPLIT_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_SPLIT_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace mlir
{
namespace Circle
{

class ConvSplit : public mlir::OpConversionPattern<mlir::ONNXSplitOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXSplitOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXSplitOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXSplitOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getInput();
    mlir::Value split = adaptor.getSplit();
    bool splitNone = mlir::isa<mlir::NoneType>(split.getType());

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvSplit name: " << op_name << "\n"; });

    int64_t axis = adaptor.getAxis();
    LLVM_DEBUG({ llvm::dbgs() << "ConvSplit axis: " << axis << "\n"; });

    mlir::Value size_splits;
    uint32_t num_splits = 0;

    if (splitNone)
    {
      // V13 SplitOp second input split is optional, and there is no input to get number of splits.
      // we can only assume from number of outputs and split from output shape.
      auto intype = mlir::dyn_cast<RankedTensorType>(input.getType());
      int64_t a_axis = axis;
      if (a_axis < 0)
        a_axis += intype.getRank();

      std::vector<int32_t> split_vals;
      auto outtypes = op.getOutputs().getTypes();
      for (size_t i = 0; i < outtypes.size(); ++i)
      {
        auto outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(outtypes[i]);
        if (outtype)
        {
          auto shape = outtype.getShape();
          split_vals.push_back(static_cast<int32_t>(shape[a_axis]));
        }
        else
          return mlir::failure();
        num_splits++;
      }
      size_splits = CreateI32Const(rewriter, split_vals, op_name + "/size");
    }
    else
    {
      size_splits = CreateI32Const(rewriter, split, op_name + "/size");
      if (mlir::isa<mlir::NoneType>(size_splits.getType()))
        return mlir::failure();

      auto sptype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(split.getType());
      auto spshape = sptype.getShape();
      assert(spshape.size() == 1);
      num_splits = spshape[0];
    }

    mlir::Value split_dim = CreateI32Const(rewriter, axis, op_name + "/axis");

    // NOTE SplitV has multiple outputs, each output can have different shape
    rewriter.replaceOpWithNewOp<SplitVOp>(op, op.getOutputs().getTypes(), input, size_splits,
                                          split_dim, num_splits);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_SPLIT_OP_H__
