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

#include "mir/ops/SqueezeOp.h"

namespace mir
{
namespace ops
{

void SqueezeOp::inferOutputTypes()
{
  assert(getNumInputs() == 1);

  const auto &input_shape = getInputShape(0);
  auto dt = getInput(0)->getElementType();
  int32_t input_rank = input_shape.rank();

  std::vector<int32_t> dims_to_squeeze;

  if (getNumSqueezeDims() == 0)
  {
    for (int32_t i = 0; i < input_rank; ++i)
    {
      if (input_shape.dim(i) == 1)
      {
        dims_to_squeeze.push_back(i);
      }
    }
  }
  else
  {
    dims_to_squeeze = getDimsToSqueeze();
    sort(dims_to_squeeze.begin(), dims_to_squeeze.end());
    dims_to_squeeze.erase(unique(dims_to_squeeze.begin(), dims_to_squeeze.end()),
                          dims_to_squeeze.end());
  }

  if (dims_to_squeeze.size() == static_cast<size_t>(input_rank))
  {
    // Input shape have 1s in all dimensions, output shape is (1,)
    setOutputType(0, {dt, Shape{1}});
    return;
  }

  int32_t output_rank = 0;
  size_t squeezing_idx = 0;
  Shape output_shape(input_rank - dims_to_squeeze.size());
  for (int32_t i = 0; i < input_rank; ++i)
  {
    if (squeezing_idx < dims_to_squeeze.size() && i == dims_to_squeeze[squeezing_idx])
    {
      if (input_shape.dim(i) != 1)
        throw std::invalid_argument("All squeezed dimensions should have size 1");

      squeezing_idx++;
    }
    else
    {
      output_shape.dim(output_rank++) = input_shape.dim(i);
    }
  }

  setOutputType(0, {dt, output_shape});
}

} // namespace ops
} // namespace mir
