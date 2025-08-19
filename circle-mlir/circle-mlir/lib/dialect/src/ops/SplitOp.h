/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
// from tensorflow/lite/kernels/internal/reference/reference_ops.h

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
  uint32_t num_splits = op.getNumSplits();
  if (op.getNumResults() != static_cast<unsigned>(num_splits))
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
  int64_t num_splits64 = static_cast<int64_t>(num_splits);
  RankedTensorType expected_output_type =
    SubstituteRankedTensorTypeDimSize(input_type, split_dim, dim_size / num_splits64);
  return VerifySplitOpOutputTypes(op.getOperation(), num_splits64,
                                  [expected_output_type](int64_t) { return expected_output_type; });
}

template <typename T>
bool splitFPTensor(Value input, int64_t split_dim, int64_t num_splits,
                   RankedTensorType expected_output_type, SmallVectorImpl<OpFoldResult> &results)
{
  assert(num_splits > 0);

  std::vector<T> input_values;
  if (!getAsConstant(input, input_values))
    return false;

  auto input_type = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto num_elements = input_type.getNumElements() / num_splits;
  assert(num_elements == expected_output_type.getNumElements());

  // prepare allocated outputs
  std::vector<std::vector<T>> output_all;
  for (int64_t i = 0; i < num_splits; ++i)
  {
    std::vector<T> output_data;
    output_data.resize(num_elements);
    output_all.push_back(output_data);
  }

  // NOTE below is reference implementation from
  //      tensorflow/lite/kernels/internal/reference/reference_ops.h
  /*
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  // For all output arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i) {
    base_inner_size *= input_shape.Dims(i);
  }

  const Scalar* input_ptr = input_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < outputs_count; ++i) {
      const int copy_size = output_shapes[i]->Dims(axis) * base_inner_size;
      memcpy(output_data[i] + k * copy_size, input_ptr,
             copy_size * sizeof(Scalar));
      input_ptr += copy_size;
    }
  }
  */

  int64_t rank = input_type.getRank();
  int64_t outer_size = 1;
  for (int64_t i = 0; i < split_dim; ++i)
  {
    outer_size *= input_type.getDimSize(i);
  }

  int64_t base_inner_size = 1;
  for (int64_t i = split_dim + 1; i < rank; ++i)
  {
    base_inner_size *= input_type.getDimSize(i);
  }

  int64_t copy_size = expected_output_type.getDimSize(split_dim) * base_inner_size;
  int64_t src = 0;
  for (int64_t k = 0; k < outer_size; k++)
  {
    for (int64_t i = 0; i < num_splits; ++i)
    {
      std::vector<T> &split_values = output_all[i];
      for (int64_t j = 0; j < copy_size; ++j)
      {
        split_values[k * copy_size + j] = input_values[src++];
      }
    }
  }

  for (int64_t i = 0; i < num_splits; ++i)
  {
    std::vector<T> &split_values = output_all[i];
    auto elements = DenseFPElementsAttr::get(expected_output_type, split_values);
    results.push_back(elements);
  }
  return true;
}

LogicalResult SplitOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results)
{
  // NOTE current implementation is for ONNXConv2D with group that converts to
  // SplitOp-[Conv2D]-Concat and this method is to fold ConstOp(filter)-SplitOp.
  SplitOp op = *this;
  Value input = op.getValue();
  {
    auto dim_type = mlir::cast<ShapedType>(op.getSplitDim().getType());
    if (!dim_type.hasStaticShape())
      return failure();
    auto input_type = mlir::cast<ShapedType>(input.getType());
    if (!input_type.hasStaticShape())
      return failure();
  }

  auto split_dim_opt = ExtractConstantIntFromTensor(op.getSplitDim());
  if (!split_dim_opt)
    return failure();

  auto input_type = mlir::dyn_cast<RankedTensorType>(input.getType());
  auto split_dim = split_dim_opt.value();
  auto rank = input_type.getRank();
  if (split_dim < 0)
    split_dim += rank;
  if (split_dim < 0 || split_dim >= rank)
    return failure();

  auto input_split_dim = input_type.getDimSize(split_dim);
  auto num_splits = static_cast<int64_t>(op.getNumSplits());
  if (num_splits == 0 || (input_split_dim % num_splits != 0))
    return failure();

  // get results shape
  auto split_dim_size = input_split_dim / num_splits;
  RankedTensorType expected_output_type =
    SubstituteRankedTensorTypeDimSize(input_type, split_dim, split_dim_size);
  LLVM_DEBUG({ llvm::dbgs() << "SplitOp::fold " << expected_output_type << "\n"; });

  auto in_etype = input_type.getElementType();
  if (in_etype.isF32())
  {
    if (!splitFPTensor<float>(input, split_dim, num_splits, expected_output_type, results))
      return failure();
  }
  else if (in_etype.isF64())
  {
    if (!splitFPTensor<double>(input, split_dim, num_splits, expected_output_type, results))
      return failure();
  }
  else
    return failure();

  return success();
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_SPLIT_OP_H__
