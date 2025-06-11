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

#ifndef __CIRCLE_MLIR_PASS_OPS_INSTANCE_NORMALIZATION_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_INSTANCE_NORMALIZATION_OP_H__

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

class ConvInstanceNormalization
  : public mlir::OpConversionPattern<mlir::ONNXInstanceNormalizationOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXInstanceNormalizationOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXInstanceNormalizationOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXInstanceNormalizationOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getInput();
    mlir::Value scale = adaptor.getScale();
    mlir::Value B = adaptor.getB();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());

    // TODO support other ranks for I/O

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    CHECK_VALID_RANK_3_4(intype);

    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    CHECK_VALID_RANK_3_4(outtype);

    auto epsilon = adaptor.getEpsilon();

    if (intype.getRank() == 4)
    {
      mlir::Value pre_tran = CreatePreTranspose(rewriter, input, op_name);

      auto output_type = GetChnLastType(outtype);
      // NOTE ONNX does not have activation so it's always NONE for Circle
      mlir::Value instnorm =
        rewriter.create<InstanceNormOp>(opLoc, output_type, pre_tran, scale, B, epsilon, "NONE");

      assert(output_type.getRank() == 4);
      ReplaceOpWithPostTranspose(rewriter, op, instnorm, op.getType(), op_name);
    }
    else
    {
      // NOTE ONNX does not have activation so it's always NONE for Circle
      rewriter.replaceOpWithNewOp<InstanceNormOp>(op, outtype, input, scale, B, epsilon, "NONE");
    }

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_INSTANCE_NORMALIZATION_OP_H__
