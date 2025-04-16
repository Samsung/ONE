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

#ifndef __CIRCLE_MLIR_PASS_OPT_FUSE_FULLYCONNECTED_ADD_H__
#define __CIRCLE_MLIR_PASS_OPT_FUSE_FULLYCONNECTED_ADD_H__

namespace mlir
{
namespace Circle
{

// Find sequence
//    FullyConnected(Nobias) - Add
// Relace with
//    FullyConnected(Bias)
struct FuseFullyConnectedAdd : public OpRewritePattern<AddOp>
{
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp add_op, PatternRewriter &rewriter) const override
  {
    mlir::Operation *is_fc = add_op.getOperand(0).getDefiningOp();
    mlir::Operation *is_const = add_op.getOperand(1).getDefiningOp();
    bool bis_fc = mlir::isa_and_nonnull<FullyConnectedOp>(is_fc);
    bool bis_const = mlir::isa_and_nonnull<ConstOp>(is_const);
    if (!mlir::isa_and_nonnull<FullyConnectedOp>(is_fc) ||
        !mlir::isa_and_nonnull<ConstOp>(is_const))
    {
      is_fc = add_op.getOperand(1).getDefiningOp();
      is_const = add_op.getOperand(0).getDefiningOp();
      bis_fc = mlir::isa_and_nonnull<FullyConnectedOp>(is_fc);
      bis_const = mlir::isa_and_nonnull<ConstOp>(is_const);
      if (!mlir::isa_and_nonnull<FullyConnectedOp>(is_fc) ||
          !mlir::isa_and_nonnull<ConstOp>(is_const))
        return mlir::failure();
    }

    auto fc_op = cast<FullyConnectedOp>(is_fc);
    auto const_op = cast<ConstOp>(is_const);
    // skip if FC already has bias
    // NOTE we can add existing bias values with values from ADD op
    // TODO implement when fc_bias is valid constant
    auto fc_bias = fc_op.getBias();
    if (!mlir::isa<mlir::NoneType>(fc_bias.getType()))
      return mlir::failure();
    // skip if FC activation is NOT none
    if (fc_op.getFusedActivationFunction() != ACT_NONE)
      return mlir::failure();

    // check constant is scalar or 1D
    bool is_const_scalar = false;
    auto const_type = mlir::cast<TensorType>(const_op.getType());
    if (const_type.getRank() == 0)
      is_const_scalar = true;
    else if (const_type.getRank() != 1)
      return mlir::failure();

    // fc attributes
    auto fc_input = fc_op.getInput();
    auto fc_filter = fc_op.getFilter();
    auto fc_wei = fc_op.getWeightsFormatAttr();
    auto fc_asy = fc_op.getAsymmetricQuantizeInputsAttr();
    auto fc_kdi = fc_op.getKeepNumDimsAttr();
    // activation from Add
    auto add_act = add_op.getFusedActivationFunctionAttr();

    mlir::Location opLoc = add_op->getLoc();

    auto op_name = GetOperationName(fc_op.getOperation());
    auto fused_loc = rewriter.getFusedLoc({fc_op.getLoc(), add_op.getLoc()});

    // if bias has one element, treat as scalar to do broadcasting
    if (const_type.getRank() == 1)
    {
      auto const_shape = const_type.getShape();
      assert(const_shape[0] > 0);
      if (const_shape[0] == 1)
        is_const_scalar = true;
    }

    if (is_const_scalar)
    {
      mlir::Value cop = const_op;
      std::vector<float> bias_values;
      if (!ExtractConstantValues(cop, bias_values))
        return mlir::failure();

      auto fc_type = mlir::cast<ShapedType>((*fc_op.getOutput().begin()).getType());
      auto fc_shape = fc_type.getShape();
      auto fc_rank = fc_type.getRank();
      int64_t num_ele = fc_shape[fc_rank - 1];
      for (int64_t c = 0; c < num_ele - 1; ++c)
        bias_values.push_back(bias_values[0]);

      auto ctt = const_type.getElementType();
      auto const_type = RankedTensorType::get({num_ele}, ctt);
      mlir::Location bias_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/bias"));
      auto const_op =
        rewriter.create<ConstOp>(bias_loc, mlir::DenseFPElementsAttr::get(const_type, bias_values));

      auto fc_op = rewriter.replaceOpWithNewOp<FullyConnectedOp>(
        add_op, add_op.getType(), fc_input, fc_filter, const_op, add_act, fc_wei, fc_kdi, fc_asy);
      fc_op->setLoc(fused_loc);
    }
    else
    {
      auto fc_op = rewriter.replaceOpWithNewOp<FullyConnectedOp>(
        add_op, add_op.getType(), fc_input, fc_filter, const_op, add_act, fc_wei, fc_kdi, fc_asy);
      fc_op->setLoc(fused_loc);
    }

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPT_FUSE_FULLYCONNECTED_ADD_H__
