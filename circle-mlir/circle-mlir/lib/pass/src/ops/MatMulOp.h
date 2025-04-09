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

#ifndef __CIRCLE_MLIR_PASS_OPS_MAT_MUL_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_MAT_MUL_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvMatMul : public mlir::OpConversionPattern<mlir::ONNXMatMulOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXMatMulOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXMatMulOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXMatMulOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getA();
    mlir::Value filter = adaptor.getB();

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType filtertype = filter.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvMatMul name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvMatMul intype: " << intype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvMatMul filtertype: " << filtertype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvMatMul outtype: " << outtype << "\n"; });

    CHECK_VALID_RANK_ATLEAST(intype, 2);
    CHECK_VALID_RANK_ATLEAST(filtertype, 2);
    CHECK_VALID_RANK_ATLEAST(outtype, 2);

    auto fr = filtertype.getRank();
    bool createFC = false;
    if (intype)
    {
      auto inshape = intype.getShape();
      switch (intype.getRank())
      {
        case 2:
          createFC = true;
          break;
        case 3:
          createFC = (inshape[0] == 1 || (inshape[1] == 1 && fr == 2));
          break;
        case 4:
          createFC = (inshape[0] == 1 && inshape[1] == 1);
          break;
      }
    }

    if (createFC)
    {
      mlir::Location opLoc = op->getLoc();

      // MatMul filter shape can be IO or 1IO or 11IO
      // FC filter shape can be rank2 OI
      // --> Add ReshapeOp to shape HW, 1HW, 11HW to HW
      auto filtershape = filtertype.getShape();
      assert(2 <= fr && fr <= 4);
      llvm::SmallVector<int64_t> filter_hws;
      filter_hws.push_back(filtershape[fr - 2]); // pick last two dims
      filter_hws.push_back(filtershape[fr - 1]);

      auto ftt = filtertype.getElementType();
      mlir::RankedTensorType resft = RankedTensorType::get(filter_hws, ftt);
      mlir::Value resft_s32 = CreateI32Const(rewriter, filter_hws, op_name + "/shape");
      mlir::Location res_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/reshape"));
      mlir::Value filter_res = rewriter.create<ReshapeOp>(res_loc, resft, filter, resft_s32);

      // --> Add Transpose to change filter from IO to OI
      llvm::SmallVector<int32_t, 4> preperm{1, 0};
      mlir::Value filter_tran = CreateTranspose(rewriter, filter_res, preperm, op_name);
      // NOTE Constant + Reshape + Transpose should be constant folded

      auto none = rewriter.getStringAttr("NONE");
      auto asymmetric_quantize_inputs = rewriter.getBoolAttr(false);
      auto keep_dims = rewriter.getBoolAttr(true);
      auto weights_format = rewriter.getStringAttr("DEFAULT");

      mlir::Value biasNone = CreateNoValue(rewriter);
      rewriter.replaceOpWithNewOp<FullyConnectedOp>(op, op.getType(), input, filter_tran, biasNone,
                                                    none, weights_format, keep_dims,
                                                    asymmetric_quantize_inputs);
    }
    else
    {
      // NOTE adjoint values are by default False.
      // from https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-mat-mul
      auto adjoint_lhs = rewriter.getBoolAttr(false);
      auto adjoint_rhs = rewriter.getBoolAttr(false);
      auto asymmetric_quantize_inputs = rewriter.getBoolAttr(false);

      rewriter.replaceOpWithNewOp<BatchMatMulOp>(op, op.getType(), input, filter, adjoint_lhs,
                                                 adjoint_rhs, asymmetric_quantize_inputs);
    }

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_MAT_MUL_OP_H__
