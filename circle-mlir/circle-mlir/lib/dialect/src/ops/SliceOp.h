/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

// from tensorflow/compiler/mlir/lite/ir/tfl_ops.cc

#ifndef __CIRCLE_MLIR_DIALECT_OPS_SLICE_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_SLICE_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult SliceOp::verify()
{
  SliceOp op = *this;
  auto input_type = mlir::cast<mlir::ShapedType>(op.getInput().getType());
  auto begin_type = mlir::cast<mlir::ShapedType>(op.getBegin().getType());
  auto size_type = mlir::cast<mlir::ShapedType>(op.getSize().getType());
  if (input_type.hasStaticShape() && begin_type.hasStaticShape() && size_type.hasStaticShape())
  {
    if (input_type.getRank() != begin_type.getNumElements())
    {
      return op.emitError("begin tensor elements size is not equal to input tensor rank");
    }

    if (input_type.getRank() != size_type.getNumElements())
    {
      return op.emitError("size tensor elements size is not equal to input tensor rank");
    }
  }

  mlir::DenseIntElementsAttr begin;
  if (matchPattern(op.getBegin(), m_Constant(&begin)))
  {
    int axis = 0;
    for (const auto &begin_i : llvm::enumerate(begin))
    {
      if (begin_i.value().getSExtValue() < 0)
      {
        return op.emitError(llvm::formatv("begin[{0}] cannot be negative", axis));
      }
      axis++;
    }
  }

  mlir::DenseIntElementsAttr size;
  if (matchPattern(op.getSize(), m_Constant(&size)))
  {
    int axis = 0;
    for (const auto &size_i : llvm::enumerate(size))
    {
      if (size_i.value().getSExtValue() < -1)
      {
        return op.emitError(llvm::formatv("size[{0}] cannot be negative other than -1", axis));
      }
      axis++;
    }
  }

  if (begin && size && input_type.hasStaticShape())
  {
    for (uint64_t i = 0, end = begin.getNumElements(); i < end; i++)
    {
      int begin_i = begin.getValues<llvm::APInt>()[i].getSExtValue();
      int size_i = size.getValues<llvm::APInt>()[i].getSExtValue();
      int dim_i = input_type.getShape()[i];
      if (begin_i > dim_i)
      {
        return op.emitOpError(
          llvm::formatv("begin[{0}] cannot exceed dimension length: {1}", i, dim_i));
      }
      if (size_i >= 0 && begin_i + size_i > dim_i)
      {
        return op.emitError(
          llvm::formatv("begin[{0}] + size[{0}] cannot exceed dimension length: {1}", i, dim_i));
      }
    }
  }

  return mlir::success();
}

ConstOp NarrowDownInt64InputValuesForOp(mlir::Operation *input_op,
                                        mlir::RankedTensorType value_type, mlir::Location loc,
                                        mlir::OpBuilder *builder)
{
  if (input_op == nullptr)
    return nullptr;

  mlir::DenseIntElementsAttr attr;
  if (!matchPattern(input_op, m_Constant(&attr)))
  {
    return nullptr;
  }

  auto value_shape_type =
    mlir::Circle::GetTypeFromTensorShape(value_type.getShape(), builder->getIntegerType(32));

  mlir::SmallVector<int32_t, 4> value_i32;
  value_i32.reserve(value_type.getRank());
  for (const auto &size : attr)
  {
    value_i32.push_back(static_cast<int32_t>(size.getSExtValue()));
  }
  auto new_value_i32_attr = mlir::DenseIntElementsAttr::get(value_shape_type, value_i32);

  return builder->create<ConstOp>(loc, new_value_i32_attr);
}

// This will cast down int64 values for slice op.
// This will require the begin & size are constants.
struct CastDownInt64BeginEndToInt32 : public mlir::OpRewritePattern<SliceOp>
{
  using mlir::OpRewritePattern<SliceOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(SliceOp slice_op,
                                      mlir::PatternRewriter &rewriter) const override
  {
    auto begin = slice_op.getBegin();
    auto size = slice_op.getSize();
    auto begin_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(begin.getType());
    auto size_type = mlir::dyn_cast_or_null<mlir::RankedTensorType>(size.getType());
    auto begin_op = begin.getDefiningOp();
    auto size_op = size.getDefiningOp();

    if (begin_op == nullptr && size_op == nullptr)
      return mlir::failure();

    if (begin_type == nullptr && size_type == nullptr)
      return mlir::failure();

    // Handle begin.
    if (begin_op && begin_type && begin_type.getElementType().isInteger(64))
    {
      auto new_begin =
        NarrowDownInt64InputValuesForOp(begin_op, begin_type, slice_op.getLoc(), &rewriter);
      if (new_begin != nullptr)
      {
        slice_op.setOperand(1, new_begin);
      }
    }

    // Handle size.
    if (size_op && size_type && size_type.getElementType().isInteger(64))
    {
      auto new_size =
        NarrowDownInt64InputValuesForOp(size_op, size_type, slice_op.getLoc(), &rewriter);
      if (new_size != nullptr)
      {
        slice_op.setOperand(2, new_size);
      }
    }

    return mlir::success();
  }
};

void SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                          mlir::MLIRContext *context)
{
  results.add<CastDownInt64BeginEndToInt32>(context);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_SLICE_OP_H__
