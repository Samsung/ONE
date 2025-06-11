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

#ifndef __CIRCLE_MLIR_PASS_OPS_RESHAPE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_RESHAPE_OP_H__

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

class ConvReshape : public mlir::OpConversionPattern<mlir::ONNXReshapeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXReshapeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXReshapeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXReshapeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();
    mlir::Value shape = adaptor.getShape();

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvReshape name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReshape intype: " << intype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvReshape outtype: " << outtype << "\n"; });

    const bool is_allowed_zero = static_cast<bool>(op.getAllowzero());
    if (is_allowed_zero)
    {
      // Not yet supported Reshpae op with allowZero=true
      return mlir::failure();
    }

    // NOTE shape is INT64 for normal case but circle2circle requries INT32.
    //      conversion to INT32 is done in opt/ConvertReshapeShape32.

    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), input, shape);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_RESHAPE_OP_H__
