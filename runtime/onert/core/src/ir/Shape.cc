/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/Shape.h"

#include <cassert>
#include <functional>
#include <numeric>
#include <algorithm>

namespace onert
{
namespace ir
{

int32_t const Shape::UNSPECIFIED_DIM = -1;

FeatureShape Shape::asFeature(Layout layout) const
{
  assert(rank() == 4);

  if (layout == Layout::NHWC)
  {
    // Feature Map in NHWC layout
    //  - Dimension(0) -> Batch
    //  - Dimension(1) -> Height
    //  - Dimension(2) -> Width
    //  - Dimension(3) -> Depth
    const auto batch = dim(0);
    const auto depth = dim(3);
    const auto height = dim(1);
    const auto width = dim(2);

    return {batch, depth, height, width};
  }
  else if (layout == Layout::NCHW)
  {
    // Feature Map in NHWC layout
    //  - Dimension(0) -> Batch
    //  - Dimension(1) -> Depth
    //  - Dimension(2) -> Height
    //  - Dimension(3) -> Width
    const auto batch = dim(0);
    const auto depth = dim(1);
    const auto height = dim(2);
    const auto width = dim(3);

    return {batch, depth, height, width};
  }
  else
  {
    throw std::runtime_error("Wrong Layout");
  }
}

// Extended dimension is filled with 1.
void Shape::extendRank(int to_rank)
{
  assert(to_rank - rank() >= 0);
  _dimensions.insert(_dimensions.cbegin(), to_rank - rank(), 1);
}

uint64_t Shape::num_elements() const
{
  // if dimension is 0, it means unspecified and cannot calculate the total number of elements
  if (std::any_of(_dimensions.begin(), _dimensions.end(),
                  [](const int32_t &v) { return v == UNSPECIFIED_DIM; }))
    throw std::runtime_error("num_elements() cannot calculate when any dimension is unspecified");

  return std::accumulate(_dimensions.cbegin(), _dimensions.cend(), UINT64_C(1),
                         std::multiplies<uint64_t>());
}

Shape permuteShape(const Shape &shape, Layout frontend_layout, Layout backend_layout)
{
  assert(shape.rank() <= 4);
  Shape backend_shape{shape};
  if (shape.rank() == 4 && frontend_layout == Layout::NHWC && backend_layout == Layout::NCHW)
  {
    backend_shape.dim(1) = shape.dim(3);
    backend_shape.dim(2) = shape.dim(1);
    backend_shape.dim(3) = shape.dim(2);
  }
  else if (shape.rank() == 4 && frontend_layout == Layout::NCHW && backend_layout == Layout::NHWC)
  {
    backend_shape.dim(1) = shape.dim(2);
    backend_shape.dim(2) = shape.dim(3);
    backend_shape.dim(3) = shape.dim(1);
  }
  return backend_shape;
}

} // namespace ir
} // namespace onert
