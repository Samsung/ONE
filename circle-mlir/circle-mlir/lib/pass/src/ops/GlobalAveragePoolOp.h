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

#ifndef __CIRCLE_MLIR_PASS_OPS_GLOBAL_AVERAGE_POOL_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_GLOBAL_AVERAGE_POOL_OP_H__

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

class ConvGlobalAveragePool : public mlir::OpConversionPattern<mlir::ONNXGlobalAveragePoolOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXGlobalAveragePoolOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXGlobalAveragePoolOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXGlobalAveragePoolOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    assert(op.verify().succeeded());

    mlir::Value input = adaptor.getX();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvGlobalAveragePool name: " << op_name << "\n"; });

    mlir::RankedTensorType intype = mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvGlobalAveragePool intype: " << intype << "\n"; });

    auto keep_dims = true;

    std::vector<int32_t> axesValue{2, 3};

    mlir::Type i32 = rewriter.getI32Type();
    auto num = static_cast<int64_t>(axesValue.size());
    mlir::RankedTensorType ptype = RankedTensorType::get({num}, i32);
    mlir::Location axes_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/axes"));
    mlir::Value axes =
      rewriter.create<ConstOp>(axes_loc, DenseIntElementsAttr::get(ptype, axesValue));

    rewriter.replaceOpWithNewOp<MeanOp>(op, op.getType(), input, axes, keep_dims);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_GLOBAL_AVERAGE_POOL_OP_H__
