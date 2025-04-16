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

#ifndef __CIRCLE_MLIR_PASS_OPT_FUSE_CONV2D_RELU_H__
#define __CIRCLE_MLIR_PASS_OPT_FUSE_CONV2D_RELU_H__

namespace mlir
{
namespace Circle
{

// Find sequence
//    Conv2D(NONE) - Transpose - RELU
// Relace with
//    Conv2D(RELU) - Transpose
template <typename RELUOP, const char *ACTIVATION>
struct FuseConv2DRelu : public OpRewritePattern<RELUOP>
{
  using OpRewritePattern<RELUOP>::OpRewritePattern;

  LogicalResult matchAndRewrite(RELUOP relu_op, PatternRewriter &rewriter) const override
  {
    mlir::Operation *is_transpose = relu_op.getOperand().getDefiningOp();
    if (!mlir::isa_and_nonnull<TransposeOp>(is_transpose))
      return mlir::failure();

    auto transpose_op = cast<TransposeOp>(is_transpose);
    mlir::Operation *is_conv2d = transpose_op.getOperand(0).getDefiningOp();
    if (!mlir::isa_and_nonnull<Conv2DOp>(is_conv2d))
      return mlir::failure();

    auto conv2d_op = cast<Conv2DOp>(is_conv2d);
    if (conv2d_op.getFusedActivationFunction() != ACT_NONE)
      return mlir::failure();

    // if transpose_op is used multiple times, do not fuse
    if (!transpose_op.getOutput().hasOneUse())
      return mlir::failure();
    if (!conv2d_op.getOutput().hasOneUse())
    {
      // if conv2d_op has multiple uses, conversion had unknown problem(s)
      // make assert to find this case if any
      assert(false);
      return mlir::failure();
    }

    mlir::Location opLoc = relu_op->getLoc();

    auto op_name = GetOperationName(conv2d_op.getOperation());
    auto fused_loc = rewriter.getFusedLoc({conv2d_op.getLoc(), relu_op.getLoc()});

    auto new_act_func = rewriter.getStringAttr(ACTIVATION);
    mlir::Value conv2d = rewriter.create<Conv2DOp>(
      fused_loc, conv2d_op.getType(), conv2d_op.getInput(), conv2d_op.getFilter(),
      conv2d_op.getBias(), conv2d_op.getDilationHFactor(), conv2d_op.getDilationWFactor(),
      new_act_func, /*padding=*/"VALID", conv2d_op.getStrideH(), conv2d_op.getStrideW());

    ReplaceOpWithPostTranspose(rewriter, relu_op, conv2d, relu_op.getType(), op_name);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPT_FUSE_CONV2D_RELU_H__
