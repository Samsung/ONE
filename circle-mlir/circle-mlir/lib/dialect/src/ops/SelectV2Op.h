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

#ifndef __CIRCLE_MLIR_DIALECT_OPS_SELECT_V2_OP_H__
#define __CIRCLE_MLIR_DIALECT_OPS_SELECT_V2_OP_H__

#include "circle-mlir/dialect/CircleDialect.h"

namespace mlir
{
namespace Circle
{

//===----------------------------------------------------------------------===//
// SelectV2Op
//===----------------------------------------------------------------------===//

template <typename T>
OpFoldResult GetSelectedDenseElementAttr(const ShapedType &out_type, const std::vector<bool> &c,
                                         const std::vector<int64_t> &x,
                                         const std::vector<int64_t> &y)
{
  std::vector<T> select_values;
  for (int32_t i = 0; i < c.size(); ++i)
  {
    auto x_val = x.at((x.size() == 1) ? 0 : i);
    auto y_val = y.at((y.size() == 1) ? 0 : i);
    select_values.push_back(c.at(i) ? x_val : y_val);
  }
  auto values = ArrayRef<T>(select_values);

  return DenseElementsAttr::get(out_type, values);
}

OpFoldResult SelectV2Op::fold(FoldAdaptor adaptor)
{
  auto operands = adaptor.getOperands();
  assert(operands.size() == 3);

  SelectV2Op op = *this;

  auto in_c = op.getCondition();
  auto in_x = op.getX();
  auto in_y = op.getY();

  // check x and y are same type
  // support i64/i32 for now
  auto x_type = in_x.getType().dyn_cast_or_null<mlir::RankedTensorType>();
  auto y_type = in_y.getType().dyn_cast_or_null<mlir::RankedTensorType>();
  mlir::Type x_etype = x_type.getElementType();
  mlir::Type y_etype = y_type.getElementType();
  if (!(x_etype.isSignlessInteger(64) || x_etype.isSignlessInteger(32)))
    return {};

  if (x_etype != y_etype)
    return {};

  std::vector<bool> at_c;
  std::vector<int64_t> at_x;
  std::vector<int64_t> at_y;
  if (!getAsConstant(in_c, at_c))
    return {};
  if (!getAsConstant(in_x, at_x))
    return {};
  if (!getAsConstant(in_y, at_y))
    return {};

  ShapedType out_type = op.getOutput().getType().cast<ShapedType>();
  if (x_etype.isSignlessInteger(64))
    return GetSelectedDenseElementAttr<int64_t>(out_type, at_c, at_x, at_y);
  else if (x_etype.isSignlessInteger(32))
    return GetSelectedDenseElementAttr<int32_t>(out_type, at_c, at_x, at_y);

  return {};
}

} // namespace Circle
} // namespace mlir

#endif // __CIRCLE_MLIR_DIALECT_OPS_SELECT_V2_OP_H__
