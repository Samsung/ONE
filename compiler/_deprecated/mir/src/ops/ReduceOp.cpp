/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/ops/ReduceMeanOp.h"

namespace mir
{
namespace ops
{

void ReduceOp::inferOutputTypes()
{
  const auto &input_shape = getInputShape(0);
  const auto &reduction_dims = getReductionDims();
  Shape output_shape;

  if (getKeepDims())
  {
    output_shape = input_shape;
    for (const int dim : reduction_dims)
    {
      output_shape.dim(dim) = 1;
    }
  }
  else
  {
    // This mask contains 'true' for dimension indices that should be reduced.
    // for example, if we want to reduce 1 and 3 dimensions with total number of dimensions 4,
    // the mask will contain: [false, true, false, true].
    std::vector<bool> reduction_dims_mask(input_shape.rank(), false);
    for (auto axis : reduction_dims)
      reduction_dims_mask[axis] = true;

    std::vector<std::int32_t> out_dims;
    out_dims.reserve(input_shape.rank() - reduction_dims.size());
    for (int axis_id = 0; axis_id < input_shape.rank(); axis_id++)
    {
      if (!reduction_dims_mask[axis_id])
        out_dims.emplace_back(input_shape.dim(axis_id));
    }
    output_shape = Shape(out_dims);
  }

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
