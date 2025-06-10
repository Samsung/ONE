/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_HELPER_BATCH_MAT_MUL_PARAMS_H__
#define __NNFW_CKER_HELPER_BATCH_MAT_MUL_PARAMS_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{
struct BatchMatMulParams
{
  BatchMatMulParams(const Shape &lhs_shape, const Shape &rhs_shape)
  {
    const Shape extended_lhs_shape = Shape::ExtendedShape(5, lhs_shape);
    const Shape extended_rhs_shape = Shape::ExtendedShape(5, rhs_shape);

    batch_dim0 = broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
    batch_dim1 = broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
    batch_dim2 = broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

    lhs_ext0 = extent(extended_lhs_shape, 0);
    lhs_ext1 = extent(extended_lhs_shape, 1);
    lhs_ext2 = extent(extended_lhs_shape, 2);
    rhs_ext0 = extent(extended_rhs_shape, 0);
    rhs_ext1 = extent(extended_rhs_shape, 1);
    rhs_ext2 = extent(extended_rhs_shape, 2);

    // Set params for each matrix multiply.
    lhs_rows = extended_lhs_shape.Dims(3);
    lhs_cols = extended_lhs_shape.Dims(4);
    rhs_rows = extended_rhs_shape.Dims(3);
    rhs_cols = extended_rhs_shape.Dims(4);
    accum_depth = extended_lhs_shape.Dims(4);
  }

  int batch_dim0;
  int batch_dim1;
  int batch_dim2;
  int lhs_ext0;
  int lhs_ext1;
  int lhs_ext2;
  int rhs_ext0;
  int rhs_ext1;
  int rhs_ext2;
  int lhs_rows;
  int lhs_cols;
  int rhs_rows;
  int rhs_cols;
  int accum_depth;

private:
  // Determines which dimension is the broadcast dimension.
  int32_t broadcast_dim(int32_t lhs_dim, int32_t rhs_dim)
  {
    if (lhs_dim == rhs_dim)
      return lhs_dim;
    if (lhs_dim == 1)
      return rhs_dim;
    assert(rhs_dim == 1);
    return lhs_dim;
  };

  // Computes the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  int extent(const Shape &shape, int x)
  {
    if (shape.Dims(x) == 1)
    {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i)
    {
      prod *= shape.Dims(i);
    }
    return prod;
  };
};
} // namespace cker
} // namespace nnfw

#endif
