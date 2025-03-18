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

namespace onert::ir
{

FeatureShape Shape::asFeature() const
{
  assert(rank() == 4);

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
                  [](const int32_t &v) { return v == kUnspecifiedDim; }))
    throw std::runtime_error("num_elements() cannot calculate when any dimension is unspecified");

  return std::accumulate(_dimensions.cbegin(), _dimensions.cend(), UINT64_C(1),
                         std::multiplies<uint64_t>());
}

Shape convertShape(const Shape &shape, const PermuteType &type)
{
  assert(shape.rank() <= Shape::kMaxRank);
  Shape ret{shape};

  if (type == ir::PermuteType::COPY || shape.rank() < 4)
    return ret;

  // Permutation changing layout beyond 4-D is not supported yet
  assert(shape.rank() <= 4);

  if (type == ir::PermuteType::NHWC_TO_NCHW)
  {
    ret.dim(1) = shape.dim(3);
    ret.dim(2) = shape.dim(1);
    ret.dim(3) = shape.dim(2);
    return ret;
  }

  assert(type == ir::PermuteType::NCHW_TO_NHWC);
  ret.dim(1) = shape.dim(2);
  ret.dim(2) = shape.dim(3);
  ret.dim(3) = shape.dim(1);
  return ret;
}

} // namespace onert::ir
