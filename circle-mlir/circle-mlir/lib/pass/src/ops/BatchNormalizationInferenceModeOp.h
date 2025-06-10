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

#ifndef __CIRCLE_MLIR_PASS_OPS_BATCH_NORMALIZATION_INFERENCE_MODE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_BATCH_NORMALIZATION_INFERENCE_MODE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include "ConvertHelper.h"

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvBatchNormalizationInferenceMode
  : public mlir::OpConversionPattern<mlir::ONNXBatchNormalizationInferenceModeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXBatchNormalizationInferenceModeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXBatchNormalizationInferenceModeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXBatchNormalizationInferenceModeOp op,
                                      OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getX();
    mlir::Value scale = adaptor.getScale();
    mlir::Value bias = adaptor.getB();
    mlir::Value mean = adaptor.getMean();
    mlir::Value variance = adaptor.getVar();

    mlir::Location opLoc = op->getLoc();

    mlir::RankedTensorType intype = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    CHECK_VALID_RANK_2_4(intype);

    mlir::RankedTensorType outtype = op.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    CHECK_VALID_RANK_2_4(outtype);

    auto op_name = GetOperationName(op.getOperation());

    float epsilon_val = op.getEpsilon().convertToFloat();
    float momentum_val = op.getMomentum().convertToFloat();
    LLVM_DEBUG({ llvm::dbgs() << "BatchNormIM name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "BatchNormIM epsilon: " << epsilon_val << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "BatchNormIM momentum: " << momentum_val << "\n"; });

    // NOTE below (a bit complicated) subgraph will be shrunk to Mul-Add by constant folding

    // 0/ broadcast scale, bias, mean, variance to match 1xCx1x1
    //    only rank 4 Constant works
    if (intype.getRank() == 4)
    {
      scale = CreateConstBroadcastChn(rewriter, input, scale, op_name + "/scale");
      bias = CreateConstBroadcastChn(rewriter, input, bias, op_name + "/bias");
      mean = CreateConstBroadcastChn(rewriter, input, mean, op_name + "/mean");
      variance = CreateConstBroadcastChn(rewriter, input, variance, op_name + "/variance");
    }

    // 1/ from onnx-tensorflow
    //   running_mean = mean * momentum + mean * (1 - momentum)
    //   running_variance = variance * momentum + variance * (1 - momentum)
    //   inputs = [x, running_mean, running_variance, bias, scale]

    mlir::Value const_momentum = CreateConst(rewriter, mean, momentum_val, op_name + "/momentum");
    mlir::Value const_1_momentum =
      CreateConst(rewriter, mean, 1.0f - momentum_val, op_name + "/1_momentum");
    assert(not const_momentum.getType().isa<mlir::NoneType>());
    assert(not const_1_momentum.getType().isa<mlir::NoneType>());

    // mul_mm = mean * momentum
    mlir::Location mul_mm_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_mm"));
    mlir::Value mul_mm =
      rewriter.create<MulOp>(mul_mm_loc, mean.getType(), mean, const_momentum, "NONE");
    // mul_1_mm = mean * (1 - momentum)
    mlir::Location mul_1_mm_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_1_mm"));
    mlir::Value mul_1_mm =
      rewriter.create<MulOp>(mul_1_mm_loc, mean.getType(), mean, const_1_momentum, "NONE");
    // running_mean = mean * momentum + mean * (1 - momentum)
    mlir::Location running_mean_loc =
      mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/running_mean"));
    mlir::Value running_mean =
      rewriter.create<AddOp>(running_mean_loc, mean.getType(), mul_mm, mul_1_mm, "NONE");

    // mul_vm = variance * momentum
    mlir::Location mul_vm_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_vm"));
    mlir::Value mul_vm =
      rewriter.create<MulOp>(mul_vm_loc, mean.getType(), variance, const_momentum, "NONE");
    // mul_1_vm = variance * (1 - momentum)
    mlir::Location mul_1_vm_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_1_vm"));
    mlir::Value mul_1_vm =
      rewriter.create<MulOp>(mul_1_vm_loc, mean.getType(), variance, const_1_momentum, "NONE");
    // running_variance = variance * momentum + variance * (1 - momentum)
    mlir::Location running_variance_loc =
      mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/running_variance"));
    mlir::Value running_variance =
      rewriter.create<AddOp>(running_variance_loc, mean.getType(), mul_vm, mul_1_vm, "NONE");

    // 2/ from tensorflow to tflite
    //   multiplier = scale / sqrt(variance + epsilon)
    //   output = (x * multiplier) + (offset - mean * multiplier)
    // as new 'running_mean' is passed from onnx-tensorflow, we need to use
    //   'running_mean' instead of 'mean' and 'running_variance' instead of 'variance'

    mlir::Value const_epsilon =
      CreateConst(rewriter, running_variance, epsilon_val, op_name + "/epsilon");
    assert(not const_epsilon.getType().isa<mlir::NoneType>());
    // add_ve = variance + epsilon
    mlir::Location add_ve_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/add_ve"));
    mlir::Value add_ve = rewriter.create<AddOp>(add_ve_loc, running_variance.getType(),
                                                running_variance, const_epsilon, "NONE");
    // sqrt_ave = sqrt(variance + epsilon) = sqrt(add_ve)
    mlir::Location sqrt_ave_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/sqrt_ave"));
    mlir::Value sqrt_ave =
      rewriter.create<SqrtOp>(sqrt_ave_loc, running_variance.getType(), add_ve);

    // multiplier = scale / sqrt(variance + epsilon) = div(scale, sqrt_ave)
    mlir::Location multiplier_loc =
      mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/multiplier"));
    mlir::Value multiplier =
      rewriter.create<DivOp>(multiplier_loc, running_variance.getType(), scale, sqrt_ave, "NONE");

    // mul_imt = x * multiplier
    mlir::Location mul_imt_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_imt"));
    mlir::Value mul_imt =
      rewriter.create<MulOp>(mul_imt_loc, input.getType(), input, multiplier, "NONE");
    // mul_mmt = mean * multiplier
    mlir::Location mul_mmt_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/mul_mmt"));
    mlir::Value mul_mmt = rewriter.create<MulOp>(mul_mmt_loc, running_variance.getType(),
                                                 running_mean, multiplier, "NONE");
    // sub_omm = offset - mean * multiplier = offset - mul_mmt
    mlir::Location sub_omm_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/sub_omm"));
    mlir::Value sub_omm =
      rewriter.create<SubOp>(sub_omm_loc, running_variance.getType(), bias, mul_mmt, "NONE");

    // output = (x * multiplier) + (offset - mean * multiplier) = mul_imt + sub_omm
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), mul_imt, sub_omm, "NONE");

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_BATCH_NORMALIZATION_INFERENCE_MODE_OP_H__
