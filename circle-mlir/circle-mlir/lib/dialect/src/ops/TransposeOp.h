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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_TRANSPOSE_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_TRANSPOSE_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace
{

// Computes the permutation of a constant `input_tensor` according to `perm`.
// The function recursively traverses the dimensions of the output tensor in
// a row-major order and writes the value in the output tensor into
// `new_values`.
void ComputePermutation(ElementsAttr input_tensor, ArrayRef<int32_t> perm,
                        ArrayRef<int64_t> output_shape, int num_dimensions, int output_axis,
                        std::vector<uint64_t> *input_indices, std::vector<Attribute> *new_values)
{
  // Refer to the implementation of `Transpose` function in
  // tensorflow/lite/kernels/internal/reference/reference_ops.h
  assert(output_axis < num_dimensions);
  const int input_axis = perm[output_axis];
  for (int i = 0; i < output_shape[output_axis]; ++i)
  {
    // Update the input indices on `input_axis`.
    input_indices->at(input_axis) = i;
    // Write the value from `input_tensor` if it is the last axis or
    // recurse into the next axis.
    const bool is_last_axis = output_axis == num_dimensions - 1;
    if (is_last_axis)
    {
      new_values->push_back(input_tensor.getValues<Attribute>()[*input_indices]);
    }
    else
    {
      ComputePermutation(input_tensor, perm, output_shape, num_dimensions, output_axis + 1,
                         input_indices, new_values);
    }
  }
}

} // namespace

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  assert(operands.size() == 2);
  auto input_tensor = operands[0].dyn_cast_or_null<ElementsAttr>();
  auto perm_tensor = operands[1].dyn_cast_or_null<ElementsAttr>();
  if (!input_tensor || !perm_tensor)
    return nullptr;

  // Do not try to fold elements attr of a quant type because
  // DenseElementsAttr does not support it.
  if (!getType().cast<ShapedType>().getElementType().isSignlessIntOrFloat())
    return nullptr;

  assert(perm_tensor.getShapedType().getRank() == 1);
  const int num_dimensions = input_tensor.getShapedType().getRank();
  assert(perm_tensor.getShapedType().getNumElements() == num_dimensions);

  ArrayRef<int64_t> input_shape = input_tensor.getShapedType().getShape();
  auto output_type = getType().cast<ShapedType>();

  SmallVector<int32_t, 4> perm;
  SmallVector<int64_t, 4> output_shape;
  for (int i = 0; i < num_dimensions; ++i)
  {
    perm.push_back(perm_tensor.getValues<IntegerAttr>()[i].getInt());
    output_shape.push_back(input_shape[perm[i]]);

    // Check that the derived output shape matches the static shape.
    assert(!output_type.hasStaticShape() || output_type.getShape()[i] == output_shape[i]);
  }

  std::vector<Attribute> new_values;
  new_values.reserve(input_tensor.getShapedType().getNumElements());
  std::vector<uint64_t> input_indices(num_dimensions);
  ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                     /*output_axis=*/0, &input_indices, &new_values);
  auto result_type =
    mlir::Circle::GetTypeFromTensorShape(output_shape, output_type.getElementType());
  return DenseElementsAttr::get(result_type, new_values);
}

mlir::LogicalResult TransposeOp::verify()
{
  TransposeOp op = *this;
  auto input_type = op.getInput().getType().cast<ShapedType>();
  auto perm_type = op.getPerm().getType().cast<ShapedType>();
  auto output_type = op.getOutput().getType().cast<ShapedType>();
  if (input_type.hasStaticShape() && perm_type.hasStaticShape())
  {
    if (perm_type.getNumElements() != input_type.getRank())
    {
      return op.emitOpError("perm tensor elements size is not equal to input tensor rank");
    }
  }

  mlir::DenseIntElementsAttr perm;
  if (!matchPattern(op.getPerm(), m_Constant(&perm)))
  {
    return success();
  }

  int index = 0;
  llvm::SmallVector<int64_t, 4> axes;
  for (const auto &axis_int : perm.getValues<APInt>())
  {
    const int64_t axis = axis_int.getSExtValue();
    if (axis < 0 || (input_type.hasRank() && axis >= input_type.getRank()))
    {
      return op.emitOpError(llvm::formatv("perm[{0}] must be in [0, rank)", index));
    }
    if (std::count(axes.begin(), axes.end(), axis) > 0)
    {
      return op.emitOpError(llvm::formatv("perm[{0}] cannot have duplicated axis", index));
    }
    axes.push_back(axis);
    index++;
  }

  if (input_type.hasStaticShape() && output_type.hasStaticShape())
  {
    llvm::SmallVector<int64_t, 4> transposed_shape;
    for (int64_t axis : axes)
    {
      transposed_shape.push_back(input_type.getDimSize(axis));
    }
    auto expected_output_type =
      mlir::Circle::GetTypeFromTensorShape(transposed_shape, input_type.getElementType());
    if (failed(mlir::verifyCompatibleShape(output_type, expected_output_type)))
    {
      return op.emitOpError(
        llvm::formatv("expect output type {0}, got {1}", expected_output_type, output_type));
    }
  }

  // TODO enable quantization

  return success();
}

static void BuildTransposeOp(OpBuilder *builder, OperationState &result, Value input, Value perm)
{
  // Output size is only known if input is ranked and perm is a constant.
  auto input_type = input.getType().cast<TensorType>();
  mlir::DenseIntElementsAttr perm_const;
  if (!input_type.hasRank() || !matchPattern(perm, m_Constant(&perm_const)) || perm_const.empty())
  {
    TransposeOp::build(*builder, result, UnrankedTensorType::get(input_type.getElementType()),
                       input, perm);
    return;
  }

  const auto perm_value_it = perm_const.value_begin<APInt>();

  const ArrayRef<int64_t> input_shape = input_type.getShape();
  SmallVector<int64_t, 4> output_shape(input_shape.size());

  for (int i = 0; i < output_shape.size(); ++i)
  {
    const APInt perm_val = perm_value_it[i];
    output_shape[i] = input_shape[perm_val.getSExtValue()];
  }

  auto element_type = input_type.getElementType();

  // TODO enable quantization

  TransposeOp::build(*builder, result,
                     mlir::Circle::GetTypeFromTensorShape(output_shape, element_type), input, perm);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_TRANSPOSE_OP_H__
