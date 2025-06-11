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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_RESHAPE_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_RESHAPE_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

namespace
{

// This pattern matches and merges a cir.reshape under the following
// condition:
// * The input's defining op is another cir.reshape.
// TODO(antiagainst): This pattern probably should be moved to the peephole
// category, after we have the infra for peephole passes.
struct RemoveAdjacentReshape : public RewritePattern
{
  explicit RemoveAdjacentReshape(MLIRContext *context)
    : RewritePattern(ReshapeOp::getOperationName(), 1, context)
  {
  }

  LogicalResult match(Operation *op) const override
  {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = thisOp.getOperand(0).getDefiningOp();
    return isa_and_nonnull<ReshapeOp>(prevOp) ? success() : failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override
  {
    auto thisOp = cast<ReshapeOp>(op);
    auto prevOp = cast<ReshapeOp>(thisOp.getOperand(0).getDefiningOp());

    // Replace
    //   %1 = "cir.reshape"(%0, %shape0)
    //   %2 = "cir.reshape"(%1, %shape1)
    // With
    //   %2 = "cir.reshape"(%0, %shape1)
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, thisOp.getType(), prevOp.getOperand(0),
                                           thisOp.getOperand(1));
  }
};

// The kernel expects an 1-D tensor for the shape operand if it presents. If all
// the dimensions are '1's except the last dimension, it will be reshaped to a
// 1-D tensor.
// Note that this pattern doesn't check or change the content of the shape
// tensor.
struct ConvertShapeTo1D : public OpRewritePattern<ReshapeOp>
{
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult rewriteNegInput(ReshapeOp reshape, PatternRewriter &rewriter) const
  {
    // input shape of ReshapeOp must be static
    auto input_type = mlir::dyn_cast_or_null<ShapedType>(reshape.getOperand(0).getType());
    if (not input_type)
      return failure();
    for (int64_t dim : input_type.getShape())
    {
      if (ShapedType::isDynamic(dim))
        return failure();
    }

    mlir::DenseIntElementsAttr shape;
    if (!matchPattern(reshape.getShape(), m_Constant(&shape)))
    {
      return failure();
    }

    auto input_shape = input_type.getShape();
    llvm::SmallVector<int64_t, 4> input_sh_vals(input_shape.begin(), input_shape.end());
    int64_t in_dim_index = 0;

    SmallVector<int64_t, 4> shape_data;
    for (const auto &it : shape.getValues<APInt>())
    {
      // NOTE This is a workaround. The meaning of the value 0 varies depending on the attribute
      //      `allowzero` of Reshape op.
      //      `allowzero=0` (by default) indicates that the corresponding dimension value is
      //      copied from the input tensor dynamically.
      //      allowzero=1 indicates that if any value in the ‘shape’ input is set to zero,
      //      the zero value is honored, similar to NumPy.
      auto v = it.getSExtValue();
      if (!v)
        v = input_sh_vals[in_dim_index];
      shape_data.push_back(v);
      in_dim_index++;
    }

    if (!shape_data.empty() && shape_data.back() == -1)
    {
      int64_t inferred_dim = 1;
      auto input_type = mlir::cast<ShapedType>(reshape.getOperand(0).getType());
      for (int64_t dim : input_type.getShape())
      {
        inferred_dim *= dim;
      }
      for (int64_t i = 0; i < shape_data.size() - 1; ++i)
      {
        inferred_dim /= shape_data[i];
      }
      shape_data.back() = inferred_dim;

      SmallVector<int64_t, 4> shape_type_dim;
      shape_type_dim.push_back(shape_data.size());

      auto new_shape_type = RankedTensorType::get(shape_type_dim, rewriter.getIntegerType(64));
      auto new_value_i64_attr = mlir::DenseIntElementsAttr::get(new_shape_type, shape_data);
      auto new_output_type = RankedTensorType::get(shape_data, input_type.getElementType());

      // Replace the old shape with the new shape
      auto new_shape_op =
        rewriter.create<ConstOp>(reshape.getLoc(), new_shape_type, new_value_i64_attr);

      // Replace the reshape operation itself
      rewriter.replaceOpWithNewOp<ReshapeOp>(reshape, new_output_type, reshape.getOperand(0),
                                             new_shape_op.getResult());

      return success();
    }
    return failure();
  }

  LogicalResult matchAndRewrite(ReshapeOp reshape, PatternRewriter &rewriter) const override
  {
    if (!reshape.getShape().hasOneUse())
      return failure();

    mlir::DenseIntElementsAttr shape;
    if (!matchPattern(reshape.getShape(), m_Constant(&shape)))
    {
      return failure();
    }
    // It is already a 1-D constant
    auto old_shape = shape.getType().getShape();
    if (old_shape.size() == 1)
    {
      // Check if the last shape value is -1, then replace it with positive value.
      return rewriteNegInput(reshape, rewriter);
    }
    // Verify all the leading dimensions are length one, except the last one.
    for (auto it = ++old_shape.rbegin(); it != old_shape.rend(); ++it)
    {
      if (*it != 1)
      {
        reshape->emitError("Non-vector shape input is used, might cause runtime error");
        return failure();
      }
    }
    auto new_shape = shape.reshape(
      GetTypeFromTensorShape({*old_shape.rbegin()}, shape.getType().getElementType()));
    rewriter.replaceOpWithNewOp<ConstOp>(reshape.getShape().getDefiningOp(), new_shape);
    return success();
  }
};

} // namespace

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  // Remove identity reshape with both static result and input shape.
  auto result_type = mlir::cast<ShapedType>(getType());
  auto input_type = mlir::cast<ShapedType>(getOperand(0).getType());
  if (InputOutputHasSameShape(input_type, result_type))
    return getInput();

  // Constant folding
  if (auto dense_elements = mlir::dyn_cast_or_null<DenseElementsAttr>(operands[0]))
  {
    // If the result type isn't static, tries to derive the result type from
    // the #2 operand.
    if (!result_type.hasStaticShape())
    {
      auto shape_elements = mlir::dyn_cast_or_null<DenseElementsAttr>(operands[1]);
      if (!shape_elements)
        return nullptr;

      SmallVector<int64_t, 4> shape_data;
      for (const auto &it : shape_elements.getValues<APInt>())
      {
        shape_data.push_back(it.getSExtValue());
      }
      result_type = GetTypeFromTensorShape(shape_data, input_type.getElementType());
    }
    return dense_elements.reshape(result_type);
  }

  return nullptr;
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<RemoveAdjacentReshape, ConvertShapeTo1D>(context);
}

using ReshapeErrorHandler = llvm::function_ref<LogicalResult(const llvm::Twine &)>;

LogicalResult GetReshapeOutputType(Value input, Value shape, ReshapeErrorHandler error_handler,
                                   TensorType &output_ty)
{
  auto input_ty = mlir::cast<TensorType>(input.getType());
  auto element_ty = input_ty.getElementType();
  output_ty = UnrankedTensorType::get(element_ty);

  auto shape_ty = mlir::dyn_cast<RankedTensorType>(shape.getType());
  if (!shape_ty)
    return success();
  if (shape_ty.getRank() != 1)
    return error_handler(
      llvm::formatv("requires 'shape' to be rank 1, but got {0}", shape_ty.getRank()));

  mlir::DenseIntElementsAttr shape_attr;
  if (!matchPattern(shape, m_Constant(&shape_attr)))
  {
    // If only shape of `shape` is known, return ranked but dynamic output
    // shape.
    if (shape_ty.hasStaticShape())
    {
      llvm::SmallVector<int64_t, 8> dynamic_shape(shape_ty.getDimSize(0), ShapedType::kDynamic);
      output_ty = GetTypeFromTensorShape(dynamic_shape, element_ty);
    }
    return success();
  }

  auto input_shape = input_ty.getShape();
  llvm::SmallVector<int64_t, 4> input_sh_vals(input_shape.begin(), input_shape.end());
  int64_t in_dim_index = 0;

  // Detect if reshape output shape is folded.
  int unknown_index = -1;
  // The product of constant shape argument excluding unknown dimension.
  int64_t shape_ty_size = 1;
  llvm::SmallVector<int64_t, 8> output_ty_shape;
  output_ty_shape.reserve(shape_attr.getNumElements());
  for (const auto &dim : llvm::enumerate(shape_attr.getValues<APInt>()))
  {
    int64_t size = dim.value().getSExtValue();
    if (size == kDynamicSize || size == ShapedType::kDynamic)
    {
      if (unknown_index != -1)
        return error_handler(
          llvm::formatv("requires 'shape' to have at most one dynamic dimension, but got "
                        "multiple dynamic dimensions at indices {0} and {1}. You need to "
                        "set up the unspecified size(s) to avoid this problem, for example,"
                        "setting batch size in keras model or setting unspecified input "
                        "size(s) with fixed ones.",
                        unknown_index, dim.index()));

      unknown_index = dim.index();
    }
    else if (size == 0)
    {
      size = input_sh_vals[in_dim_index];
      shape_ty_size *= size;
    }
    else if (size > 0)
    {
      shape_ty_size *= size;
    }
    else
    {
      return error_handler(llvm::formatv("requires 'shape' to have dimensions greater than -1, "
                                         "but got {0} at index {1}",
                                         size, dim.index()));
    }
    output_ty_shape.push_back(size);
    in_dim_index++;
  }

  if (!input_ty.hasStaticShape())
  {
    output_ty = GetTypeFromTensorShape(output_ty_shape, element_ty);
    return success();
  }

  // Compute the value of the unknown dimension.
  if (unknown_index != -1)
  {
    // Compute number of elements in tensor shape.
    int64_t input_ty_size = 1;
    for (const auto &dim : input_ty.getShape())
    {
      if (dim > 0)
      {
        input_ty_size *= dim;
      }
      else if (dim < 0)
      {
        return error_handler(llvm::formatv("Reshape got invalid dim size {0}", dim));
      }
    }

    const int64_t missing_dim = input_ty_size / shape_ty_size;
    if (shape_ty_size * missing_dim != input_ty_size)
      return error_handler(llvm::formatv("requires 'input' number of elements be a multiple of "
                                         "{0}, but got {1}",
                                         shape_ty_size, input_ty_size));

    // Set the unknown dimension such that total number of elements remain
    // constant.
    output_ty_shape[unknown_index] = missing_dim;
  }

  output_ty = GetTypeFromTensorShape(output_ty_shape, element_ty);

  return success();
}

mlir::LogicalResult ReshapeOp::verify()
{
  ReshapeOp op = *this;
  auto error_handler = [&op](const llvm::Twine &message) -> LogicalResult {
    return op.emitOpError() << message;
  };
  TensorType expected_ty;
  if (failed(GetReshapeOutputType(op.getInput(), op.getShape(), error_handler, expected_ty)))
    return failure();

  auto output_ty = mlir::dyn_cast<RankedTensorType>(op.getType());
  if (!output_ty)
    return success();
  auto input_ty = mlir::cast<TensorType>(op.getInput().getType());
  if (output_ty.hasStaticShape() && input_ty.hasStaticShape())
  {
    const int64_t output_ty_size = output_ty.getNumElements();
    const int64_t input_ty_size = input_ty.getNumElements();
    if (input_ty_size != output_ty_size)
      return op.emitOpError() << "requires 'output' number of elements to "
                                 "match 'input' number of elements, but got "
                              << output_ty_size << " and " << input_ty_size;
  }

  /* TODO enable AreCastCompatible if necessary
  if (!AreCastCompatible({output_ty, expected_ty}))
    return op.emitOpError()
           << "requires 'output' type " << output_ty
           << " to be cast compatible with expected type " << expected_ty;
  */
  return success();
}

LogicalResult ReshapeOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                          ValueRange operands, DictionaryAttr attr,
                                          OpaqueProperties properties, RegionRange,
                                          SmallVectorImpl<Type> &inferredReturnTypes)
{
  ReshapeOpAdaptor op(operands, attr, properties);
  const Value input = op.getInput();
  const Value shape = op.getShape();

  auto error_handler = [&](const llvm::Twine &message) -> LogicalResult {
    // A dummy error handler.
    // Errors when computing the output shape will be raised in
    // ReshapeOp::verify call.
    return failure();
  };
  TensorType output_type;
  if (GetReshapeOutputType(input, shape, error_handler, output_type).succeeded())
  {
    inferredReturnTypes.assign({output_type});
    return success();
  }
  Type result_type;
  result_type = UnrankedTensorType::get(mlir::cast<ShapedType>(input.getType()).getElementType());
  inferredReturnTypes.assign({result_type});
  return success();
}

namespace
{

// from llvm-project/mlir/lib/IR/TypeUtilities.cpp with some modification
bool verifyCompatibleShape0as1(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2)
{
  if (shape1.size() != shape2.size())
    return false;
  for (auto dims : llvm::zip(shape1, shape2))
  {
    int64_t dim1 = std::get<0>(dims);
    int64_t dim2 = std::get<1>(dims);
    if (!ShapedType::isDynamic(dim1) && !ShapedType::isDynamic(dim2) && dim1 != dim2)
    {
      if (dim1 == 0 and dim2 == 1 || dim1 == 1 and dim2 == 0)
        continue;
      return false;
    }
  }
  return true;
}

} // namespace

bool ReshapeOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs)
{
  if (lhs.size() != rhs.size() || lhs.size() != 1)
    return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0])))
  {
    // lhs is inferredReturnTypes
    // rhs is returnTypes (from source onnx)
    // we need to check if returnTypes has 0 then it is compatible if inferredReturnTypes is 1
    auto sType1 = llvm::dyn_cast<ShapedType>(lhs[0]);
    auto sType2 = llvm::dyn_cast<ShapedType>(rhs[0]);
    return verifyCompatibleShape0as1(sType1.getShape(), sType2.getShape());
  }
  return true;
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_RESHAPE_OP_H__
