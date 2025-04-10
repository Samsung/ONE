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

#ifndef __CIRCLE_MLIR_PASS_OPT_CONVERT_DIV_ERF_TO_MUL_ERF_H__
#define __CIRCLE_MLIR_PASS_OPT_CONVERT_DIV_ERF_TO_MUL_ERF_H__

#include <cmath>

namespace mlir
{
namespace Circle
{

// Find sequence
//    Div(X,C)-Custom(Erf), where C is Constant
// Relace Div with Mul as
//    Mul(X,1.0f/C)-Custom(Erf)
struct ConvertDivErfToMulErf : public OpRewritePattern<CustomOp>
{
  using OpRewritePattern<CustomOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomOp custom_op, PatternRewriter &rewriter) const override
  {
    if (custom_op.getCustomCode().str() != "Erf")
      return mlir::failure();
    if (custom_op.getNumOperands() != 1u)
      return mlir::failure();

    mlir::Operation *is_div = custom_op.getOperand(0).getDefiningOp();
    if (!mlir::isa_and_nonnull<DivOp>(is_div))
      return mlir::failure();
    auto div_op = cast<DivOp>(is_div);

    mlir::Operation *is_const = div_op.getOperand(1).getDefiningOp();
    if (!mlir::isa_and_nonnull<ConstOp>(is_const))
      return mlir::failure();

    auto const_op = cast<ConstOp>(is_const);
    auto const_type = const_op.getType().cast<TensorType>();
    if (not const_type.getElementType().isF32())
      return mlir::failure();

    mlir::Value cop = const_op;
    std::vector<float> cop_values;
    if (!ExtractConstantValues(cop, cop_values))
      return mlir::failure();

    auto const_shape = const_type.getShape();
    int64_t numElements = 1;
    for (size_t dim = 0; dim < const_shape.size(); ++dim)
      numElements = numElements * const_shape[dim];
    for (int64_t c = 0; c < numElements; ++c)
    {
      if (cop_values[c] == 0.0f || std::isnan(cop_values[c]))
      {
        LLVM_DEBUG({ llvm::dbgs() << "ConvertDivErfToMulErf failed: divide by 0.0f or NaN\n"; });
        return mlir::failure();
      }
      cop_values[c] = 1.0f / cop_values[c];
    }

    auto inv_const_op = rewriter.create<ConstOp>(
      is_const->getLoc(), mlir::DenseFPElementsAttr::get(const_type, cop_values));
    auto div_act = div_op.getFusedActivationFunction();
    rewriter.replaceOpWithNewOp<MulOp>(div_op, div_op.getType(), div_op.getLhs(), inv_const_op,
                                       div_act);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPT_CONVERT_DIV_ERF_TO_MUL_ERF_H__
