/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/ops/TransposeOp.h"

namespace mir
{
namespace ops
{

TransposeOp::TransposeOp(Output *arg, const std::vector<std::size_t> &axis_order)
  : Operation(Type::transpose, {arg}), _axis_order(axis_order)
{
  assert(_axis_order.size() == static_cast<std::size_t>(getInputShape(0).rank()));
  inferOutputTypes();
}

void TransposeOp::inferOutputTypes()
{
  auto &input_shape = getInputShape(0);
  Shape output_shape(input_shape.rank());
  for (std::size_t i = 0; i < _axis_order.size(); ++i)
    output_shape.dim(static_cast<std::int64_t>(i)) =
      input_shape.dim(static_cast<int32_t>(_axis_order.at(i)));

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
