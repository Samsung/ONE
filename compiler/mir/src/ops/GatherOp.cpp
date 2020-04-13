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

#include "mir/ops/GatherOp.h"

namespace mir
{
namespace ops
{

void GatherOp::inferOutputTypes()
{
  const auto &data_shape = getInputShape(0);
  const auto &indices_shape = getInputShape(1);

  auto data_rank = data_shape.rank();
  auto indices_rank = indices_shape.rank();
  auto output_rank = data_rank + indices_rank - 1;

  assert(_axis >= -data_rank && _axis < data_rank);
  int32_t axis = _axis < 0 ? _axis + data_rank : _axis;

  Shape output_shape;
  output_shape.resize(output_rank);

  // Output shape is data.shape[:axis] + indices.shape + data.shape[axis + 1:].
  int32_t output_index = 0;
  for (int32_t i = 0; i < axis; ++i)
    output_shape.dim(output_index++) = data_shape.dim(i);
  for (int32_t i = 0; i < indices_rank; ++i)
    output_shape.dim(output_index++) = indices_shape.dim(i);
  for (int32_t i = axis + 1; i < data_rank; ++i)
    output_shape.dim(output_index++) = data_shape.dim(i);

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
