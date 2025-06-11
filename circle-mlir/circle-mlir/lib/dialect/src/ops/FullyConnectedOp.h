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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_FULLYCONNECTED_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_FULLYCONNECTED_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// FullyConnectedOp
//===----------------------------------------------------------------------===//

LogicalResult FullyConnectedOp::verify()
{
  FullyConnectedOp op = *this;
  ShapedType input_type = mlir::cast<ShapedType>(op.getInput().getType());
  ShapedType filter_type = mlir::cast<ShapedType>(op.getFilter().getType());
  if (filter_type.hasRank() && filter_type.getRank() != 2)
  {
    return op.emitOpError("expect 2d filter, got ") << filter_type;
  }

  if (!input_type.hasStaticShape() || !filter_type.hasStaticShape())
  {
    return mlir::success();
  }

  // Input's element size must be multiple of parameter's z_in dimension.
  const int z_in = filter_type.getDimSize(1);
  const int num_input_elements = input_type.getNumElements();
  if (z_in != 0 && num_input_elements % z_in != 0)
  {
    return op.emitOpError(
             llvm::formatv("expect 'input' num_elements % {0} == 0, got input type ", z_in))
           << input_type;
  }

  // TODO(jpienaar): Include more shape verification for SHUFFLED4x16INT8
  // format.
  if (op.getWeightsFormat() == "DEFAULT")
  {
    ShapedType output_type = mlir::cast<ShapedType>((*op.getOutput().begin()).getType());
    if (!output_type.hasStaticShape())
    {
      return mlir::success();
    }

    const int num_output_elements = output_type.getNumElements();
    const int z_out = filter_type.getDimSize(0);
    if (num_output_elements % z_out != 0)
    {
      return op.emitOpError(llvm::formatv("expect 'output' num_elements % {0} == 0, got ", z_out))
             << output_type;
    }

    if (z_in != 0 && num_input_elements / z_in != num_output_elements / z_out)
    {
      return op.emitOpError("num_input_elements / z_in != num_output_elements / z_out");
    }
  }

  return mlir::success();
}

LogicalResult FullyConnectedOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results)
{
  auto operands = adaptor.getOperands();
  assert(operands.size() == 3);

  // Folding not implemented with any activation function or any weight type
  // besides the default.
  if (getFusedActivationFunction() != "NONE")
    return failure();
  if (getWeightsFormat() != "DEFAULT")
    return failure();

  // Bias tensor is optional.
  const bool has_bias = !(!getBias() || mlir::isa<NoneType>(getBias().getType()));

  // Get the tensors.
  DenseElementsAttr input_tensor, weights_tensor, bias_tensor;
  if (!matchPattern(getInput(), m_Constant(&input_tensor)) ||
      !matchPattern(getFilter(), m_Constant(&weights_tensor)) ||
      (has_bias && !matchPattern(getBias(), m_Constant(&bias_tensor))))
  {
    return failure();
  }

  // Get the tensor types.
  const auto input_type = mlir::cast<ShapedType>(input_tensor.getType());
  const auto weights_type = mlir::cast<ShapedType>(weights_tensor.getType());
  const auto bias_type = has_bias ? mlir::cast<ShapedType>(bias_tensor.getType()) : ShapedType{};

  const auto output_type = mlir::cast<ShapedType>(getType(0));

  // Folding only implemented for float tensors.
  if (!input_type.getElementType().isF32() || !weights_type.getElementType().isF32() ||
      !output_type.getElementType().isF32() || (has_bias && !bias_type.getElementType().isF32()))
  {
    return failure();
  }

  // Folding only implemented for static shapes
  if (!input_type.hasStaticShape() || !weights_type.hasStaticShape() ||
      (has_bias && !bias_type.hasStaticShape()))
  {
    return failure();
  }

  // Folding only implemented for 1D input, 2D weights and 1D bias
  if (input_type.getShape().size() != 1 || weights_type.getShape().size() != 2 ||
      (has_bias && bias_type.getShape().size() != 1))
  {
    return failure();
  }

  // Get the sizes
  const auto input_size = input_type.getNumElements();
  const auto output_size = output_type.getNumElements();

  // Get iterators to the tensors.
  const auto input_values_it = input_tensor.getValues<float>().begin();
  const auto weights_values_ptr = weights_tensor.getValues<float>().begin();
  auto weights_row_it = weights_values_ptr;
  // The 'else' case could be nullptr, but the types don't match.
  auto bias_values_it = has_bias ? bias_tensor.getValues<float>().begin() : input_values_it;

  // Do the actual folding, one output at a time.
  std::vector<float> result_values;
  result_values.reserve(output_size);

  for (int i = 0; i < output_size; ++i)
  {
    // Dot product with Kahan/Neumaier summation to minimize numeric errors.
    float sum = has_bias ? *bias_values_it : 0.0f;
    float compensation = 0.0f;
    for (int j = 0; j < input_size; ++j)
    {
      const float addend = input_values_it[j] * weights_row_it[j];
      const float new_sum = sum + addend;
      // DO NOT enable -funsafe-math-optimizations here.
      // There is a test detecting unsafe optimizations.
      // Unsafe math optimizations can reorder float formulas, and set the
      // compensation to constant 0. The formula must be evaluated as written
      // for the algorithm to work.
      // (Note: -ffast-math is a superset of -funsafe-math-optimizations.)
      if (std::abs(sum) >= std::abs(addend))
      {
        compensation += (sum - new_sum) + addend;
      }
      else
      {
        compensation += (addend - new_sum) + sum;
      }
      sum = new_sum;
    }
    result_values.push_back(sum + compensation);
    weights_row_it += input_size;
    bias_values_it++;
  }

  // Set result tensor
  const auto folded = DenseElementsAttr::get(output_type, ArrayRef<float>(result_values));
  results.assign({folded});

  return success();
}

void FullyConnectedOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<RemoveOptionalZeroBias<FullyConnectedOp>>(context);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_FULLYCONNECTED_OP_H__
