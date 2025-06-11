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

#ifndef __CIRCLE_MLIR_PASS_OPS_SQUEEZE_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_SQUEEZE_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <cassert>
#include <stdexcept>
#include <vector>

namespace mlir
{
namespace Circle
{

class ConvSqueeze : public mlir::OpConversionPattern<mlir::ONNXSqueezeOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXSqueezeOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXSqueezeOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXSqueezeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getData();
    mlir::Value axes = adaptor.getAxes();

    mlir::Location opLoc = op->getLoc();
    auto op_name = GetOperationName(op.getOperation());

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    mlir::RankedTensorType outtype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvSqueeze name: " << op_name << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvSqueeze intype: " << intype << "\n"; });
    LLVM_DEBUG({ llvm::dbgs() << "ConvSqueeze outtype: " << outtype << "\n"; });

    std::vector<int64_t> values;
    if (!ExtractConstantValues(axes, values))
      return mlir::failure();

    auto axesArray = rewriter.getI64ArrayAttr(values);

    rewriter.replaceOpWithNewOp<SqueezeOp>(op, op.getType(), input, axesArray);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_SQUEEZE_OP_H__
