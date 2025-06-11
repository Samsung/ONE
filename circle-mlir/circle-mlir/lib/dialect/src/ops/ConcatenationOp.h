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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_CONCATENATION_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_CONCATENATION_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// ConcatenationOp
//===----------------------------------------------------------------------===//
// TODO(ashwinm): Implement shape inference for Concatenation

namespace
{

int64_t GetConcatenationOpAxis(ConcatenationOp op)
{
  auto output_type = mlir::cast<RankedTensorType>(op.getOutput().getType());
  int32_t axis = op.getAxis();
  if (axis < 0)
    axis += output_type.getRank();
  return axis;
}

// Verify operand types and the result type:
//
// 1. Operand type ranks must be equal to the output type rank.
//
// 2. Operand dimension sizes (except dimension `axis`) must be equal to
//    previously seen dimension sizes of the same dimension.
//
// 3. Sum of operand dimension sizes of the `axis` dimension must be equal to
//    the dimension size of the `axis` dimension of output.
//
// Note: If an operand has unranked tensor type or has dynamic dimension size,
// those dimensions will be skipped.
LogicalResult VerifyConcatenationOpTypes(Operation *op, RankedTensorType output_type,
                                         ArrayRef<TensorType> operand_types, int64_t axis)
{
  const int64_t output_rank = output_type.getRank();

  SmallVector<int64_t, 4> result_dim_sizes_loc(output_rank, ShapedType::kDynamic);
  SmallVector<int64_t, 4> result_dim_sizes(output_type.getShape().begin(),
                                           output_type.getShape().end());
  result_dim_sizes[axis] = 0;

  auto FormatLoc = [&result_dim_sizes_loc](int64_t dim) {
    const int64_t loc = result_dim_sizes_loc[dim];
    if (loc == ShapedType::kDynamic)
      return std::string("output");
    return llvm::formatv("operand #{0}", loc).str();
  };

  for (const auto &operand : llvm::enumerate(operand_types))
  {
    auto operand_type = mlir::dyn_cast<RankedTensorType>(operand.value());
    if (!operand_type)
    {
      result_dim_sizes[axis] = ShapedType::kDynamic;
      continue;
    }

    const int64_t operand_rank = operand_type.getRank();
    if (operand_rank != output_rank)
      return op->emitOpError() << "rank of operand #" << operand.index()
                               << " must be equal to rank of output, expected " << output_rank
                               << ", got " << operand_rank;

    for (int64_t dim = 0; dim < output_rank; ++dim)
    {
      const int64_t operand_dim_size = operand_type.getDimSize(dim);
      const int64_t result_dim_size = result_dim_sizes[dim];

      if (dim == axis)
      {
        if (ShapedType::isDynamic(operand_dim_size) || ShapedType::isDynamic(result_dim_size))
        {
          result_dim_sizes[axis] = ShapedType::kDynamic;
        }
        else
        {
          result_dim_sizes[axis] += operand_dim_size;
        }
        continue;
      }

      if (ShapedType::isDynamic(operand_dim_size))
        continue;

      if (ShapedType::isDynamic(result_dim_size))
      {
        result_dim_sizes[dim] = operand_dim_size;
        result_dim_sizes_loc[dim] = operand.index();
        continue;
      }

      if (result_dim_size != operand_dim_size)
        return op->emitOpError() << "dimension size of dimension #" << dim << " of operand #"
                                 << operand.index() << " must be equal to "
                                 << "dimension size of dimension #" << dim << " of "
                                 << FormatLoc(dim) << ", expected " << result_dim_size << ", got "
                                 << operand_dim_size;
    }
  }

  const int64_t output_concated_dim_size = output_type.getDimSize(axis);
  if (!ShapedType::isDynamic(output_concated_dim_size) &&
      !ShapedType::isDynamic(result_dim_sizes[axis]) &&
      result_dim_sizes[axis] != output_concated_dim_size)
    return op->emitOpError() << "dimension size of dimension #" << axis << " of output "
                             << "must be equal to the sum of dimension sizes of dimension #" << axis
                             << ", expected " << result_dim_sizes[axis] << ", got "
                             << output_concated_dim_size;

  return success();
}

// Returns true when all operands are instances of DenseElementsAttr and the
// output type has a static shape.
bool IsConcatenationOpConstFoldable(ConcatenationOp op, ArrayRef<Attribute> operands,
                                    RankedTensorType output_type, int64_t axis)
{
  if (operands.empty())
    return false;
  if (!output_type.hasStaticShape())
    return false;
  if (axis < 0)
    return false;

  return llvm::all_of(
    operands, [](Attribute operand) { return operand && mlir::isa<DenseElementsAttr>(operand); });
}

DenseElementsAttr ConstFoldConcatenateOpDense(ArrayRef<Attribute> operands,
                                              RankedTensorType output_type, int64_t axis)
{
  const auto outer_dims = output_type.getShape().take_front(axis);
  const int64_t outer_size =
    std::accumulate(outer_dims.begin(), outer_dims.end(), 1, std::multiplies<int64_t>());

  const auto base_inner_dims = output_type.getShape().drop_front(axis + 1);
  const int64_t base_inner_size =
    std::accumulate(base_inner_dims.begin(), base_inner_dims.end(), 1, std::multiplies<int64_t>());

  // Splits each input operand into outer_size pieces and combines them in
  // round-robin ordering.
  std::vector<Attribute> out_attrs(output_type.getNumElements());
  int64_t out = 0;
  for (int64_t outer = 0; outer < outer_size; ++outer)
  {
    for (auto op : operands)
    {
      auto typed_attr = mlir::cast<TypedAttr>(op);
      const int64_t dim_size = mlir::cast<RankedTensorType>(typed_attr.getType()).getDimSize(axis);
      const int64_t inner_size = dim_size * base_inner_size;

      auto input_attrs = mlir::cast<DenseElementsAttr>(op).getValues<Attribute>();
      auto input_iter = input_attrs.begin() + outer * inner_size;
      for (int64_t inner = 0; inner < inner_size; ++inner)
        out_attrs[out++] = *input_iter++;
    }
  }

  return DenseElementsAttr::get(output_type, out_attrs);
}

} // end anonymous namespace

LogicalResult ConcatenationOp::verify()
{
  ConcatenationOp op = *this;
  auto output_type = mlir::dyn_cast<RankedTensorType>(op.getOutput().getType());

  // If the output type is unranked, there is nothing else to be verified.
  if (!output_type)
    return success();

  const int64_t axis = GetConcatenationOpAxis(op);
  if (axis < 0 || axis >= output_type.getRank())
    return op.emitOpError("concatenation dimension must be in [-rank, rank)");

  SmallVector<TensorType, 4> operand_types;
  for (Value operand : op.getValues())
    operand_types.push_back(mlir::cast<TensorType>(operand.getType()));

  return VerifyConcatenationOpTypes(op.getOperation(), output_type, operand_types, axis);
}

OpFoldResult ConcatenationOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  if (getFusedActivationFunction() == "NONE")
  {
    if (auto output_type = mlir::dyn_cast<RankedTensorType>(getOutput().getType()))
    {
      const int64_t axis = GetConcatenationOpAxis(*this);
      if (IsConcatenationOpConstFoldable(*this, operands, output_type, axis))
        return ConstFoldConcatenateOpDense(operands, output_type, axis);
    }
  }

  // Remove all empty values.
  SmallVector<Value, 4> non_empty_values;
  for (Value value : this->getValues())
  {
    const auto shaped_type = mlir::cast<ShapedType>(value.getType());
    if (shaped_type.hasStaticShape() && shaped_type.getNumElements() == 0)
    {
      continue;
    }
    non_empty_values.push_back(value);
  }

  // All are not empty, do nothing.
  if (non_empty_values.size() == getNumOperands())
    return nullptr;

  // If only one input is non-empty, just return it as the result of folding.
  if (non_empty_values.size() == 1)
  {
    return non_empty_values[0];
  }

  // Otherwise, build a new concatenation op with non-empty values.
  mlir::OpBuilder builder(getOperation());
  auto new_concat = builder.create<ConcatenationOp>(
    getLoc(), getType(), non_empty_values,
    builder.getIntegerAttr(builder.getIntegerType(32), static_cast<int64_t>(getAxis())),
    builder.getStringAttr(getFusedActivationFunction()));
  return new_concat.getResult();
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_CONCATENATION_OP_H__
