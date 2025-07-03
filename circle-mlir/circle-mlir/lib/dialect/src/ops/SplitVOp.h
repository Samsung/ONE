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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_SPLIT_V_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_SPLIT_V_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// SplitVOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult SplitVOp::verify()
{
  SplitVOp op = *this;
  int64_t num_splits = static_cast<int64_t>(op.getNumSplits());
  if (static_cast<int64_t>(op.getNumResults()) != num_splits)
    return op.emitOpError("output count should match 'num_splits' attribute");

  // If 'split_dim' is not a constant, there are no other checks.
  std::optional<int64_t> split_dim_opt = ExtractConstantIntFromTensor(op.getSplitDim());
  if (!split_dim_opt)
    return success();

  // If 'input' is not a ranked tensor, there are no other checks.
  auto input_type = mlir::dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!input_type)
    return success();

  int64_t split_dim = split_dim_opt.value();
  const int64_t rank = input_type.getRank();
  if (split_dim < 0)
    split_dim += rank;
  if (split_dim < 0 || split_dim >= rank)
    return op.emitOpError("'split_dim' should be in [-rank, rank)");

  // If the 'split_dim' dimension of the 'input' tensor has a dynamic size,
  // there are no other checks.
  const int64_t dim_size = input_type.getDimSize(split_dim);
  if (ShapedType::isDynamic(dim_size))
    return success();

  // If 'size_splits' is not a constant, there are no other checks.
  ElementsAttr size_splits_attr;
  if (!matchPattern(op.getSizeSplits(), m_Constant(&size_splits_attr)))
    return success();

  if (static_cast<int64_t>(size_splits_attr.getNumElements()) != num_splits)
  {
    auto size_splits_type = mlir::cast<RankedTensorType>(op.getSizeSplits().getType());
    RankedTensorType expected_size_splits_type =
      mlir::Circle::GetTypeFromTensorShape({num_splits}, size_splits_type.getElementType());
    return op.emitOpError("'size_splits' should be ") << expected_size_splits_type;
  }

  // Normalizes and verifies 'size_splits'.
  // Note: TensorFlow allows one -1 element in 'size_splits'.  The -1 element
  // means the rest of the dimension size.
  llvm::SmallVector<int64_t, 4> size_splits;
  size_splits.reserve(num_splits);

  int64_t negative_size_split_loc = -1;
  int64_t total_size_splits = 0;

  for (int64_t i = 0; i < num_splits; ++i)
  {
    auto size_split_attr = size_splits_attr.getValues<IntegerAttr>()[i];
    int64_t size_split = size_split_attr.getValue().getSExtValue();
    size_splits.push_back(size_split);
    if (size_split >= 0)
    {
      total_size_splits += size_split;
      continue;
    }
    if (size_split < -1)
      return op.emitOpError("elements of 'size_splits' should be greater than or equal to -1");
    if (negative_size_split_loc != -1)
      return op.emitOpError("'size_splits' can only have one -1");
    negative_size_split_loc = i;
  }

  if (negative_size_split_loc != -1)
  {
    if (total_size_splits > dim_size)
      return op.emitOpError("sum of non-negative elements of 'size_splits' is greater than the "
                            "dimension size of 'split_dim' axis");
    size_splits[negative_size_split_loc] = dim_size - total_size_splits;
    total_size_splits = dim_size;
  }

  if (total_size_splits != dim_size)
    return op.emitOpError("sum of 'size_splits' should match the dimension size of 'split_dim' "
                          "axis");

  // Verifies result tensor types.
  auto get_expected_output_type = [input_type, split_dim, &size_splits](int64_t i) {
    return SubstituteRankedTensorTypeDimSize(input_type, split_dim, size_splits[i]);
  };
  return VerifySplitOpOutputTypes(op.getOperation(), num_splits, get_expected_output_type);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_SPLIT_V_OP_H__
