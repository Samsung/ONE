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

#ifndef __CIRCLE_MLIR_PASS_OPS_ARG_MAX_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_ARG_MAX_OP_H__

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

class ConvArgMax : public mlir::OpConversionPattern<mlir::ONNXArgMaxOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXArgMaxOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXArgMaxOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXArgMaxOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());

    mlir::Location opLoc = op->getLoc();

    int64_t axis = adaptor.getAxis();

    auto op_name = GetOperationName(op.getOperation());

    LLVM_DEBUG({ llvm::dbgs() << "ConvArgMax name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvArgMax axis: " << axis << "\n"; });

    if (notYetImplemented(adaptor))
      return mlir::failure();

    mlir::Value inpAxis = CreateI32Const(rewriter, axis, op_name + "/dim");

    rewriter.replaceOpWithNewOp<ArgMaxOp>(op, op.getType(), input, inpAxis);

    return mlir::success();
  }

private:
  bool notYetImplemented(OpAdaptor adaptor) const
  {
    int64_t keepdims = adaptor.getKeepdims();
    int64_t select_last_index = adaptor.getSelectLastIndex();

    LLVM_DEBUG({ llvm::dbgs() << "ConvArgMax keepdims: " << keepdims << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvArgMax select_last_index: " << select_last_index << "\n"; });

    if (keepdims != 0)
      return true;

    if (select_last_index != 0)
      return true;

    return false;
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_ARG_MAX_OP_H__
