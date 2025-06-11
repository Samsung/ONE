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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_SHAPE_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_SHAPE_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// ShapeOp
//===----------------------------------------------------------------------===//

OpFoldResult ShapeOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  auto input_type = mlir::cast<ShapedType>(getInput().getType());
  if (!input_type.hasStaticShape())
    return nullptr;

  ArrayRef<int64_t> shape = input_type.getShape();
  auto result_type = mlir::cast<ShapedType>(getType());
  if (result_type.getElementType().isInteger(64))
  {
    return DenseElementsAttr::get<int64_t>(result_type, shape);
  }
  else if (result_type.getElementType().isInteger(32))
  {
    SmallVector<int32_t, 4> shape_i32;
    shape_i32.reserve(shape.size());
    for (int64_t dim : shape)
    {
      shape_i32.push_back(dim);
    }
    return DenseElementsAttr::get<int32_t>(result_type, shape_i32);
  }
  return nullptr;
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_SHAPE_OP_H__
