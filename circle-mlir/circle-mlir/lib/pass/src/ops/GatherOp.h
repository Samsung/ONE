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

#ifndef __CIRCLE_MLIR_PASS_OPS_GATHER_OP_H__
#define __CIRCLE_MLIR_PASS_OPS_GATHER_OP_H__

#include <circle-mlir/dialect/CircleDialect.h>

#include "ConvertHelper.h"

#include <mlir/IR/Matchers.h> // from @llvm-project
#include <mlir/Transforms/DialectConversion.h>

#include <src/Dialect/ONNX/ONNXOps.hpp>

#include <vector>

namespace mlir
{
namespace Circle
{

class ConvGather : public mlir::OpConversionPattern<mlir::ONNXGatherOp>
{
public:
  using mlir::OpConversionPattern<mlir::ONNXGatherOp>::OpConversionPattern;
  using OpAdaptor = typename mlir::ONNXGatherOp::Adaptor;

  mlir::LogicalResult matchAndRewrite(mlir::ONNXGatherOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override
  {
    mlir::Value input = adaptor.getData();
    mlir::Value indices = adaptor.getIndices();
    const auto axis = adaptor.getAxis();

    mlir::Location opLoc = op->getLoc();

    auto op_name = GetOperationName(op.getOperation());
    LLVM_DEBUG({ llvm::dbgs() << "ConvGather name: " << op_name << "\n"; });

    // Q) How to correct invalid index values?
    // A) element-wise floormod(add(x, y), y) = (x + y) - floor((x + y) / y) * y, if x is an element
    // and y is size of dimension to gather
    mlir::RankedTensorType ranked_input_type =
      mlir::dyn_cast_or_null<mlir::RankedTensorType>(input.getType());
    LLVM_DEBUG({ llvm::dbgs() << "ConvGather ranked_input_type: " << ranked_input_type << "\n"; });

    // Assume input have shape
    const auto y = ranked_input_type.getShape()[axis];
    mlir::RankedTensorType ranked_indices_type =
      mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
    LLVM_DEBUG(
      { llvm::dbgs() << "ConvGather ranked_indices_type: " << ranked_indices_type << "\n"; });
    auto element_type = ranked_indices_type.getElementType();

    std::vector<int64_t> indices_values;
    if (ExtractConstantValues(indices, indices_values))
    {
      for (auto &&e : indices_values)
      {
        const auto &x = e;
        e = (x + y) - ((x + y) / y) * y;
      }

      mlir::Location indices_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/indices"));

      if (element_type.isInteger(32))
      {
        llvm::SmallVector<int32_t, 8> v32;
        v32.reserve(indices_values.size());
        for (int64_t v : indices_values)
        {
          if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max())
            return rewriter.notifyMatchFailure(op, "index out of int32 range");
          v32.push_back(static_cast<int32_t>(v));
          auto const_type =
            mlir::RankedTensorType::get(ranked_indices_type.getShape(), rewriter.getI32Type());
          indices =
            rewriter.create<ConstOp>(indices_loc, mlir::DenseIntElementsAttr::get(const_type, v32));
        }
      }
      else if (element_type.isInteger(64))
      {
        auto const_type =
          mlir::RankedTensorType::get(ranked_indices_type.getShape(), rewriter.getI64Type());
        indices = rewriter.create<ConstOp>(
          indices_loc, mlir::DenseIntElementsAttr::get(const_type, indices_values));
      }
      else
        return rewriter.notifyMatchFailure(op, "indices element type is not i32/i64");
    }
    else
    {
      // Add operators that correct invalid indices values to valid indices values if indices is not
      // constant
      mlir::RankedTensorType ranked_indices_type =
        mlir::dyn_cast_or_null<mlir::RankedTensorType>(indices.getType());
      mlir::RankedTensorType scalar_type =
        mlir::RankedTensorType::get({}, ranked_indices_type.getElementType());
      mlir::Location y_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/y"));
      mlir::Value const_y =
        rewriter.create<ConstOp>(y_loc, mlir::DenseIntElementsAttr::get(scalar_type, {y}));
      mlir::Location add_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/add"));
      mlir::Value pre_add =
        rewriter.create<AddOp>(add_loc, indices.getType(), indices, const_y, "NONE");
      mlir::Location indices_loc = mlir::NameLoc::get(rewriter.getStringAttr(op_name + "/indices"));
      indices = rewriter.create<FloorModOp>(indices_loc, indices.getType(), pre_add, const_y);
    }

    // Assume output type is same as input type
    rewriter.replaceOpWithNewOp<GatherOp>(op, op.getType(), input, indices, axis);

    return mlir::success();
  }
};

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_PASS_OPS_GATHER_OP_H__
