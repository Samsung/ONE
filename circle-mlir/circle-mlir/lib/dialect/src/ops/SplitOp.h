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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_SPLIT_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_SPLIT_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult SplitOp::verify()
{
  SplitOp op = *this;
  int64_t num_splits = op.getNumSplits();
  if (op.getNumResults() != num_splits)
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

  if (dim_size % num_splits != 0)
    return op.emitOpError("'num_splits' should evenly divide 'split_dim' axis");

  // Verifies output tensor types.
  RankedTensorType expected_output_type =
    SubstituteRankedTensorTypeDimSize(input_type, split_dim, dim_size / num_splits);
  return VerifySplitOpOutputTypes(op.getOperation(), num_splits,
                                  [expected_output_type](int64_t) { return expected_output_type; });
}

LogicalResult SplitOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results)
{
  // TODO implement

  return failure();
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_SPLIT_OP_H__
