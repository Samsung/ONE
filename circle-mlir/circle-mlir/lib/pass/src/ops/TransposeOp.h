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

#ifndef __CIRCLE_MLIR_PASS_OPS_TRANSPOSE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_TRANSPOSE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>

namespace mlir
{
namespace Circle
{

class ConvTranspose : public mlir::OpConversionPattern<mlir::ONNXTransposeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXTransposeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXTransposeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXTransposeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvTranspose name: " << op_name << "\n"; });

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvTranspose intype: " << intype << "\n"; });

    auto inshape = intype.getShape();

    std::vector<int32_t> permValue;
    if (!getPerm(op, permValue))
    {
      LLVM_DEBUG({ llvm::dbgs() << "ConvTranspose NO perm()!\n"; });
      // fill in int sequence from 0
      for (size_t i = 0; i < inshape.size(); ++i)
        permValue.push_back(static_cast<int32_t>(i));
    }
    mlir::Value perm = CreateI32Const(rewriter, permValue, op_name + "/perm");

    rewriter.replaceOpWithNewOp<TransposeOp>(op, op.getType(), input, perm);

    return mlir::success();
  }

private:
  bool getPerm(mlir::ONNXTransposeOp &op, std::vector<int32_t> &values) const
  {
    auto perm = op.getPerm();
    if (perm.has_value())
    {
      auto value = perm.value();
      for (int i = 0; i < value.size(); ++i)
      {
        auto v = GetIntValue<int32_t>(value, i);
        values.push_back(v);
      }
      return true;
    }
    return false;
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_TRANSPOSE_OP_H__
