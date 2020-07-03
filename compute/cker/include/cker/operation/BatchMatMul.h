/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_BATCH_MATMUL_H__
#define __NNFW_CKER_BATCH_MATMUL_H__

#include "Transpose.h"

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"
#include "cker/operation/reference/BatchMatMul.h"

#include <vector>

namespace nnfw
{
namespace cker
{

class BatchMatMul
{
public:
  BatchMatMul()
  {
    // DO NOTHING
  }

  /**
   * @brief   Prepare temporary area for calculation
   */
  void prepare(const Shape &lhs_shape, const Shape &rhs_shape, bool adj_x, bool adj_y)
  {
    if (adj_x)
    {
      int32_t rank = lhs_shape.DimensionsCount();
      _temp_lhs_shape.Resize(rank);

      for (int32_t i = 0; i < rank - 2; i++)
      {
        _temp_lhs_shape.SetDim(i, lhs_shape.Dims(i));
      }
      _temp_lhs_shape.SetDim(rank - 2, lhs_shape.Dims(rank - 1));
      _temp_lhs_shape.SetDim(rank - 1, lhs_shape.Dims(rank - 2));

      _temp_lhs.resize(_temp_lhs_shape.FlatSize());
    }

    if (!adj_y)
    {
      int32_t rank = rhs_shape.DimensionsCount();
      _temp_rhs_shape.Resize(rank);

      for (int32_t i = 0; i < rank - 2; i++)
      {
        _temp_rhs_shape.SetDim(i, rhs_shape.Dims(i));
      }
      _temp_rhs_shape.SetDim(rank - 2, rhs_shape.Dims(rank - 1));
      _temp_rhs_shape.SetDim(rank - 1, rhs_shape.Dims(rank - 2));

      _temp_rhs.resize(_temp_rhs_shape.FlatSize());
    }
  }

  void operator()(const Shape &lhs_shape, const float *lhs_data, const Shape &rhs_shape,
                  const float *rhs_data, bool adj_x, bool adj_y, const Shape &output_shape,
                  float *output_data)
  {
    // Assume lhs and rhs is not constant
    // TODO Handle constant input

    if (!adj_y)
    {
      transposeRowsCols(rhs_shape, rhs_data, _temp_rhs_shape, _temp_rhs.data());
    }

    if (adj_x)
    {
      transposeRowsCols(lhs_shape, lhs_data, _temp_lhs_shape, _temp_lhs.data());
    }

    Shape new_lhs_shape = adj_x ? lhs_shape : swapRowColDims(lhs_shape);
    Shape new_rhs_shape = adj_y ? rhs_shape : swapRowColDims(rhs_shape);
    const float *new_lhs_data = adj_x ? _temp_lhs.data() : lhs_data;
    const float *new_rhs_data = adj_y ? rhs_data : _temp_rhs.data();

    // Note we pass RHS args first, LHS args second
    // Check accumulative dimensions of lhs and rhs of are equal
    assert(Shape::ExtendedShape(5, new_rhs_shape).Dims(4) ==
           Shape::ExtendedShape(5, new_lhs_shape).Dims(3));
    reference::BatchMatMul(new_rhs_shape, new_rhs_data, new_lhs_shape, new_lhs_data, output_shape,
                           output_data);
  }

private:
  Shape swapRowColDims(const Shape &shape)
  {
    Shape swapped_shape(shape);
    const uint32_t dims = shape.DimensionsCount();
    swapped_shape.SetDim(dims - 2, shape.Dims(dims - 1));
    swapped_shape.SetDim(dims - 1, shape.Dims(dims - 2));

    return swapped_shape;
  }

  void transposeRowsCols(const Shape &input_shape, const float *input_data,
                         const Shape &output_shape, float *output_data)
  {
    TransposeParams params;
    int rank = input_shape.DimensionsCount();
    params.perm_count = rank;
    for (int i = 0; i < 2; i++)
    {
      params.perm[i] = i;
    }
    params.perm[rank - 2] = rank - 1;
    params.perm[rank - 1] = rank - 2;

    Transpose<float>(params, input_shape, input_data, output_shape, output_data);
  }

private:
  std::vector<float> _temp_lhs;
  Shape _temp_lhs_shape;
  std::vector<float> _temp_rhs;
  Shape _temp_rhs_shape;
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_BATCH_MATMUL_H__
