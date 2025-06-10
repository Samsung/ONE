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

#ifndef __CIRCLE_MLIR_PASS_OPT_CONVERT_SQRT_DIV_TO_RSQRT_H__
#define __CIRCLE_MLIR_PASS_OPT_CONVERT_SQRT_DIV_TO_RSQRT_H__

#include <cmath>

namespace mlir
{
namespace Circle
{

// Find sequence
//    Sqrt(X)-Div(1.0,S), where S is output of Sqrt(X)
// Relace with Rsqrt()
//    Rsqrt(X)
struct ConvertSqrtDivToRsqrt : public OpRewritePattern<DivOp>
{
  using OpRewritePattern<DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOp div_op, PatternRewriter &rewriter) const override
  {
    mlir::Operation *is_const = div_op.getOperand(0).getDefiningOp();
    if (!mlir::isa_and_nonnull<ConstOp>(is_const))
      return mlir::failure();

    mlir::Operation *is_sqrt = div_op.getOperand(1).getDefiningOp();
    if (!mlir::isa_and_nonnull<SqrtOp>(is_sqrt))
      return mlir::failure();

    auto const_op = cast<ConstOp>(is_const);
    auto const_type = mlir::cast<TensorType>(const_op.getType());
    if (not const_type.getElementType().isF32())
      return mlir::failure();

    mlir::Value cop = const_op;
    std::vector<float> cop_values;
    if (!ExtractConstantValues(cop, cop_values))
      return mlir::failure();

    for (const auto value : cop_values)
    {
      if (value != 1.0f || std::isnan(value))
        return mlir::failure();
    }

    auto sqrt_op = cast<SqrtOp>(is_sqrt);
    mlir::Value sqrt_input = sqrt_op.getOperand();

    rewriter.replaceOpWithNewOp<RsqrtOp>(div_op, div_op.getType(), sqrt_input);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPT_CONVERT_SQRT_DIV_TO_RSQRT_H__
