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

#ifndef __CIRCLE_MLIR_PASS_OPS_GEMM_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_GEMM_OP_H__

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

class ConvGemm : public mlir::OpConversionPattern<mlir::ONNXGemmOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXGemmOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXGemmOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXGemmOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvGemm name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvGemm alpha: " << op.getAlpha().convertToFloat() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvGemm beta: " << op.getBeta().convertToFloat() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvGemm transA: " << op.getTransA() << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvGemm transB: " << op.getTransB() << "\n"; });

    // out = alpha * (input ​@ mat2) + beta * bias
    // bias' = bias * beta
    // mat2' = mat2 * alpha
    // out = (input ​@ mat2') + bias'
    // if transA is 1, add Transpose to input and also to FC mat2
    // if transB is 1, this is normal to produce FC, if 0, add Transpose to mat2
    //
    // default: alpha=1.0, beta=1.0, transA=0, transB=1

    mlir::Value input = adaptor.getA();
    mlir::Value filter = adaptor.getB();
    mlir::Value bias = adaptor.getC();
    bool biasNone = bias.getType().isa<mlir::NoneType>();
    auto alpha = op.getAlpha().convertToFloat();
    auto beta = op.getBeta().convertToFloat();
    auto transA = op.getTransA();
    auto transB = op.getTransB();

    mlir::Location opLoc = op->getLoc();

    // TODO support other ranks for I/O

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvGemm intype: " << intype << "\n"; });
    CHECK_VALID_RANK_2(intype);

    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    LLVM_DEBUG({ llvm::dbgs() << "ConvGemm outtype: " << outtype << "\n"; });
    CHECK_VALID_RANK_2(outtype);

    auto none = rewriter.getStringAttr("NONE");

    // get Mul(filter, alpha)
    mlir::Value alphaConst = CreateConst(rewriter, filter, alpha, op_name + "/alpha");
    mlir::Location ma_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_alpha"));
    mlir::Value mulfa = rewriter.create<MulOp>(ma_loc, filter.getType(), filter, alphaConst, none);

    // get Mul(bias, beta)
    mlir::Value mulbb;
    if (not biasNone)
    {
      mlir::Value betaConst = CreateConst(rewriter, bias, beta, op_name + "/beta");
      mlir::Location mb_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_beta"));
      mulbb = rewriter.create<MulOp>(mb_loc, bias.getType(), bias, betaConst, none);
    }

    if (transA != 0)
    {
      mlir::Value pre_perm_in;
      mlir::Value pre_perm_we;
      // TODO support rank 3, 4
      {
        llvm::SmallVector<int32_t, 4> pre_vals{1, 0};
        mlir::Location itp_loc =
          mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/input/tr/perm"));
        pre_perm_in = rewriter.create<ConstOp>(itp_loc, GetI32ElementsAttr(pre_vals, &rewriter));
        mlir::Location atp_loc =
          mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/alpha/tr/perm"));
        pre_perm_we = rewriter.create<ConstOp>(atp_loc, GetI32ElementsAttr(pre_vals, &rewriter));
      }
      mlir::Location it_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/input/tr"));
      mlir::Value pre_tran_in = rewriter.create<TransposeOp>(it_loc, input, pre_perm_in);
      mlir::Location at_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/alpha/tr"));
      mlir::Value pre_tran_we = rewriter.create<TransposeOp>(at_loc, mulfa, pre_perm_we);

      input = pre_tran_in;
      mulfa = pre_tran_we;
    }
    else if (transB == 0)
    {
      // Add Trapose To mulfa
      mlir::Value pre_perm;
      // TODO support rank 3, 4
      {
        llvm::SmallVector<int32_t, 4> pre_vals{1, 0};
        mlir::Location atp_loc =
          mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/alpha/tr/perm"));
        pre_perm = rewriter.create<ConstOp>(atp_loc, GetI32ElementsAttr(pre_vals, &rewriter));
      }
      mlir::Location at_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/alpha/tr"));
      mlir::Value pre_tran = rewriter.create<TransposeOp>(at_loc, mulfa, pre_perm);
      mulfa = pre_tran;
    }

    // TODO set proper boolean values
    auto asymmetric_quantize_inputs = rewriter.getBoolAttr(false);
    auto keep_dims = rewriter.getBoolAttr(false);
    auto weights_format = rewriter.getStringAttr("DEFAULT");

    mlir::Value biasZero = CreateNoValue(rewriter);
    auto fc =
      rewriter.create<FullyConnectedOp>(opLoc, op.getType(), input, mulfa, biasZero, none,
                                        weights_format, keep_dims, asymmetric_quantize_inputs);

    if (not biasNone)
    {
      mlir::Value fcres = fc->getResult(0);
      mlir::Location add_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/add"));
      auto add_op = rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), fcres, mulbb, none);
      add_op->setLoc(add_loc);
    }
    else
    {
      rewriter.replaceOp(op, fc);
    }

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_GEMM_OP_H__
