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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_CONV2D_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_CONV2D_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

void Conv2DOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  // TODO(b/180121750): Enable the pattern after the integration tests are
  // fixed.
  // results.add<RemoveOptionalZeroBias<Conv2DOp>>(context);
}

LogicalResult Conv2DOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                                         ValueRange operands, DictionaryAttr attr,
                                         OpaqueProperties properties, RegionRange,
                                         SmallVectorImpl<Type> &inferredReturnTypes)
{
  Conv2DOpAdaptor op(operands, attr, properties);

  const Value input = op.getInput();
  const Value filter = op.getFilter();

  const RankedTensorType input_ty = mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  const RankedTensorType filter_ty = mlir::dyn_cast_or_null<RankedTensorType>(filter.getType());
  // If indeed both input type & filter type are ranked type and have ranks.
  // We will need to check their ranks are valid.
  if ((input_ty && input_ty.hasRank() && input_ty.getRank() != 4) ||
      (filter_ty && filter_ty.hasRank() && filter_ty.getRank() != 4))
  {
    return emitOptionalError(location, "Invalid ranks");
  }

  // If either input or filter is unranked, we will just return unranked output
  // shape.
  if (!input_ty || !filter_ty || !input_ty.hasRank() || !filter_ty.hasRank())
  {
    Type result_type;
    result_type = UnrankedTensorType::get(mlir::cast<ShapedType>(input.getType()).getElementType());
    inferredReturnTypes.assign({result_type});
    return success();
  }

  auto stride_h = op.getStrideHAttr().getInt();
  auto stride_w = op.getStrideWAttr().getInt();
  auto dilation_h = op.getDilationHFactorAttr().getInt();
  auto dilation_w = op.getDilationWFactorAttr().getInt();

  // We don't have EXPLICIT PADDING in Circle.
  auto paddings = op.getPadding();
  mlir::Circle::Padding padding;
  auto padding_is_valid = GetPaddingFromString(paddings.str(), &padding);
  if (!padding_is_valid.ok())
  {
    return emitOptionalError(location, "invalid padding format provided");
  }

  // Output always have rank 4. All dimensions are initialized to
  // dynamic size and can be partially inferred.
  // TFL's conv2d is always NHWC format & the filter is OHWI.
  SmallVector<int64_t, 4> return_shape(4, ShapedType::kDynamic);
  return_shape[0] = input_ty.getDimSize(0);
  return_shape[3] = filter_ty.getDimSize(0);

  // Spatial dimensions can be inferred only when both input and filter are
  // ranked because we need to get their spatial dimensions.

  // Height.
  if (!input_ty.isDynamicDim(1) && !filter_ty.isDynamicDim(1))
  {
    int64_t output_height;
    if (failed(ComputeConvWindowedOutputSize(input_ty.getDimSize(1), filter_ty.getDimSize(1),
                                             dilation_h, stride_h, padding, &output_height)))
    {
      return failure();
    }
    return_shape[1] = output_height;
  }

  // Width.
  if (!input_ty.isDynamicDim(2) && !filter_ty.isDynamicDim(2))
  {
    int64_t output_width;
    if (failed(ComputeConvWindowedOutputSize(input_ty.getDimSize(2), filter_ty.getDimSize(2),
                                             dilation_w, stride_w, padding, &output_width)))
    {
      return failure();
    }
    return_shape[2] = output_width;
  }

  auto result_type = mlir::Circle::GetTypeFromTensorShape(return_shape, input_ty.getElementType());
  inferredReturnTypes.assign({result_type});

  return success();
}

bool Conv2DOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs)
{
  if (lhs.size() != rhs.size() || lhs.size() != 1)
    return false;
  if (failed(mlir::verifyCompatibleShape(lhs[0], rhs[0])))
    return false;
  return true;
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_CONV2D_OP_H__
