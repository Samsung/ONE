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

#ifndef __CIRCLE_MLIR_PASS_OPS_PAD_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_PAD_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

namespace mlir
{
namespace Circle
{

// NOTE Usually Pad is ONNXPadV2Op but shape inference is not available.
// createDecomposeONNXToONNXPass must be called to convert ONNXPadV2Op to ONNXPadOp.
class ConvPad : public mlir::OpConversionPattern<mlir::ONNXPadOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXPadOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXPadOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXPadOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value input = adaptor.getData();
    mlir::Value pads = adaptor.getPads();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvPad name: " << op_name << "\n"; });

    auto op_mode = op.getModeAttr();
    LLVM_DEBUG({ llvm::dbgs() << "ConvPad mode: " << op_mode.str() << "\n"; });

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvPad intype: " << intype << "\n"; });

    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvPad outtype: " << outtype << "\n"; });

    if (not(op_mode.str() == "reflect" || op_mode.str() == "constant"))
      return mlir::failure();

    // convert pads in 1D to 2D
    std::vector<int32_t> values;
    if (ExtractConstantValues(pads, values))
    {
      if (values.size() != 8)
        return mlir::failure();

      mlir::Type i32 = rewriter.getI32Type();
      mlir::RankedTensorType ptype = RankedTensorType::get({4, 2}, i32);
      llvm::SmallVector<int32_t, 8> pvalue = {values[0], values[4], values[1], values[5],
                                              values[2], values[6], values[3], values[7]};
      mlir::Location pv_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/padsval"));
      mlir::Value paddings =
        rewriter.create<ConstOp>(pv_loc, DenseIntElementsAttr::get(ptype, pvalue));

      if (op_mode.str() == "reflect")
      {
        auto type = mlir::Circle::MirrorPaddingType::REFLECT;
        rewriter.replaceOpWithNewOp<MirrorPadOp>(op, op.getType(), input, paddings, type);
      }
      else if (op_mode.str() == "constant")
      {
        rewriter.replaceOpWithNewOp<PadOp>(op, op.getType(), input, paddings);
      }
    }
    else
    {
      // reshape [8] to [2,4]
      auto pads_type =
        mlir::dyn_cast_or_null<mlir::RankedTensorType>(pads.getType()).getElementType();
      std::vector<int32_t> reshape_sh{2, 4};
      mlir::Value reshape_dims = CreateI32Const(rewriter, opLoc, reshape_sh);
      mlir::RankedTensorType res_type = RankedTensorType::get({2, 4}, pads_type);
      mlir::Value reshape_inp = rewriter.create<ReshapeOp>(opLoc, res_type, pads, reshape_dims);

      // transpose to [4,2]
      llvm::SmallVector<int32_t, 4> pre_vals{1, 0};
      mlir::Value pre_perm =
        rewriter.create<ConstOp>(opLoc, GetI32ElementsAttr(pre_vals, &rewriter));
      mlir::Value pre_tran = rewriter.create<TransposeOp>(opLoc, reshape_inp, pre_perm);

      if (op_mode.str() == "reflect")
      {
        auto type = mlir::Circle::MirrorPaddingType::REFLECT;
        rewriter.replaceOpWithNewOp<MirrorPadOp>(op, op.getType(), input, pre_tran, type);
      }
      else if (op_mode.str() == "constant")
      {
        rewriter.replaceOpWithNewOp<PadOp>(op, op.getType(), input, pre_tran);
      }
    }

    // NOTE with PadOp::fold(), new Circle.pad will dissappear when inshape == outshape

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_PAD_OP_H__
