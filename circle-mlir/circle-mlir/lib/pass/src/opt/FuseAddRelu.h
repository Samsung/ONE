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

#ifndef __CIRCLE_MLIR_PASS_OPT_FUSE_ADD_RELU_H__
#define __CIRCLE_MLIR_PASS_OPT_FUSE_ADD_RELU_H__

namespace mlir
{
namespace Circle
{

// Find sequence
//    Add(NONE) - RELU
// Relace with
//    Add(RELU)
template <typename RELUOP, const char *ACTIVATION>
struct FuseAddRelu : public OpRewritePattern<RELUOP>
{
  using OpRewritePattern<RELUOP>::OpRewritePattern;

  LogicalResult matchAndRewrite(RELUOP relu_op, PatternRewriter &rewriter) const override
  {
    mlir::Operation *is_add = relu_op.getOperand().getDefiningOp();
    if (!mlir::isa_and_nonnull<AddOp>(is_add))
      return mlir::failure();

    auto add_op = cast<AddOp>(is_add);
    if (add_op.getFusedActivationFunction() != ACT_NONE)
      return mlir::failure();

    auto new_act_func = rewriter.getStringAttr(ACTIVATION);

    // if add_op is used multiple times, do not fuse
    if (!add_op.getOutput().hasOneUse())
      return mlir::failure();

    auto fused_loc = rewriter.getFusedLoc({add_op.getLoc(), relu_op.getLoc()});

    auto add_op2 = rewriter.replaceOpWithNewOp<AddOp>(relu_op, relu_op.getType(), add_op.getLhs(),
                                                      add_op.getRhs(), new_act_func);

    add_op2->setLoc(fused_loc);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPT_FUSE_ADD_RELU_H__
