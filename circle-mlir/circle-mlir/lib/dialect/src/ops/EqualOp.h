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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_EQUAL_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_EQUAL_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// EqualOp
//===----------------------------------------------------------------------===//

OpFoldResult EqualOp::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  assert(operands.size() == 2);

  EqualOp op = *this;
  ShapedType out_type = op.getOutput().getType().cast<ShapedType>();
  auto in_x = op.getX();
  auto in_y = op.getY();

  // we can assume x and y are same type
  // support i64 for now
  auto x_type = in_x.getType().dyn_cast_or_null<mlir::RankedTensorType>();
  mlir::Type x_etype = x_type.getElementType();
  if (!x_etype.isSignlessInteger(64))
    return {};

  std::vector<int64_t> at_x;
  std::vector<int64_t> at_y;
  if (!getAsConstant(in_x, at_x))
    return {};
  if (!getAsConstant(in_y, at_y))
    return {};

  SmallVector<bool, 4> equal_values;
  auto num_elements = out_type.getNumElements();
  for (int64_t i = 0; i < num_elements; ++i)
  {
    auto x_val = at_x.at(i);
    auto y_val = at_y.at(i);
    equal_values.push_back(x_val == y_val ? true : false);
  }
  auto values = ArrayRef<bool>(equal_values);
  return DenseElementsAttr::get(out_type, values);
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_EQUAL_OP_H__
