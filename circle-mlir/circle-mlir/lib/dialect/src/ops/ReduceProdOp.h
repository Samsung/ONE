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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_REDUCE_PROD_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_REDUCE_PROD_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// ReduceProdOp
//===----------------------------------------------------------------------===//

OpFoldResult ReduceProdOp::fold(FoldAdaptor adaptor)
{
  auto input_type = mlir::cast<ShapedType>(getInput().getType());
  if (!input_type.hasStaticShape())
    return nullptr;
  auto axes_type = mlir::cast<ShapedType>(getAxes().getType());
  if (!axes_type.hasStaticShape())
    return nullptr;

  // TODO support for keep_dims=true
  if (getKeepDims())
    return nullptr;

  // support 1D input_type, axes_type for now
  // TODO support more ranks
  if (input_type.getRank() != 1)
    return nullptr;
  if (axes_type.getRank() != 1)
    return nullptr;
  // support single element axes_type for now
  ArrayRef<int64_t> shape = axes_type.getShape();
  if (shape.size() != 1)
    return nullptr;

  ReduceProdOp op = *this;
  auto in_input = op.getInput();
  auto in_axes = op.getAxes();

  std::vector<int64_t> inputs_v;
  std::vector<int32_t> axes_v;
  if (!getAsConstant(in_input, inputs_v))
    return nullptr;
  if (!getAsConstant(in_axes, axes_v))
    return nullptr;

  // support axis=[0] for now
  if (axes_v[0] != 0)
    return nullptr;

  // keep_dims=false and 1D input makes scalar
  int64_t result = 1;
  for (const auto &v : inputs_v)
    result *= v;

  auto result_type = mlir::cast<ShapedType>(getType());
  return DenseElementsAttr::get<int64_t>(result_type, result);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_REDUCE_PROD_OP_H__
