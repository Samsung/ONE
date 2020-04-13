/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tflite/TensorShapeUtils.h"

namespace nnfw
{
namespace tflite
{

nnfw::misc::tensor::Shape broadcast(const nnfw::misc::tensor::Shape &lhs_shape,
                                    const nnfw::misc::tensor::Shape &rhs_shape)
{
  const uint32_t lhs_rank = lhs_shape.rank();
  const uint32_t rhs_rank = rhs_shape.rank();
  const uint32_t out_rank = std::max(lhs_rank, rhs_rank);
  const uint32_t lhs_rank_diff = out_rank - lhs_rank;
  const uint32_t rhs_rank_diff = out_rank - rhs_rank;

  nnfw::misc::tensor::Shape out_shape(out_rank);

  for (uint32_t axis = 0; axis < out_rank; ++axis)
  {
    out_shape.dim(axis) = std::max(axis < lhs_rank_diff ? 1 : lhs_shape.dim(axis - lhs_rank_diff),
                                   axis < rhs_rank_diff ? 1 : rhs_shape.dim(axis - rhs_rank_diff));
  }

  return out_shape;
}

} // namespace tflite
} // namespace nnfw
