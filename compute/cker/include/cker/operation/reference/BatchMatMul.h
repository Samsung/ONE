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

#ifndef __NNFW_CKER_REFERENCE_BATCH_MATMUL_H__
#define __NNFW_CKER_REFERENCE_BATCH_MATMUL_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/operation/optimized/Gemm.h"

namespace nnfw
{
namespace cker
{
namespace reference
{

namespace impl
{
struct BMMParams
{
  BMMParams(const Shape &lhs_shape, const Shape &rhs_shape)
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

inline void bmm_reference(const BMMParams &bmm_params, const float *lhs_data, const float *rhs_data,
                          float *output_data)
{
  for (int b0 = 0; b0 < bmm_params.batch_dim0; ++b0)
  {
    const float *lhs_ptr0 = lhs_data + (b0 * bmm_params.lhs_ext0);
    const float *rhs_ptr0 = rhs_data + (b0 * bmm_params.rhs_ext0);
    for (int b1 = 0; b1 < bmm_params.batch_dim1; ++b1)
    {
      const float *lhs_ptr1 = lhs_ptr0 + b1 * bmm_params.lhs_ext1;
      const float *rhs_ptr1 = rhs_ptr0 + b1 * bmm_params.rhs_ext1;
      for (int b2 = 0; b2 < bmm_params.batch_dim2; ++b2)
      {
        const float *lhs_ptr2 = lhs_ptr1 + b2 * bmm_params.lhs_ext2;
        const float *rhs_ptr2 = rhs_ptr1 + b2 * bmm_params.rhs_ext2;
        float *out_ptr = output_data + ((b0 * bmm_params.batch_dim1 * bmm_params.batch_dim2) +
                                        b1 * bmm_params.batch_dim2 + b2) *
                                         bmm_params.lhs_rows * bmm_params.rhs_cols;
        for (int j = 0; j < bmm_params.rhs_cols; ++j)
        {
          for (int i = 0; i < bmm_params.lhs_rows; ++i)
          {
            float total = 0.f;
            for (int k = 0; k < bmm_params.accum_depth; ++k)
            {
              total +=
                lhs_ptr2[bmm_params.accum_depth * i + k] * rhs_ptr2[j * bmm_params.accum_depth + k];
            }
            int idx = bmm_params.lhs_rows * j + i;
            out_ptr[idx] = total;
          }
        }
      }
    }
  }
}

#if defined(CKER_X86_PLATFORM)
inline void bmm_optimized(const BMMParams &bmm_params, const float *lhs_data, const float *rhs_data,
                          float *output_data)
{
  MatrixParams<float> lhs_params; // should it be created from rhs?
  lhs_params.order = Order::kRowMajor;
  lhs_params.rows = bmm_params.lhs_rows;
  lhs_params.cols = bmm_params.lhs_cols;
  lhs_params.cache_policy = nnfw::cker::optimized::DefaultCachePolicy(false);

  MatrixParams<float> rhs_params;
  rhs_params.order = Order::kRowMajor;
  rhs_params.rows = bmm_params.rhs_rows;
  rhs_params.cols = bmm_params.rhs_cols;
  rhs_params.cache_policy = nnfw::cker::optimized::DefaultCachePolicy(false);

  MatrixParams<float> dst_params;
  dst_params.order = Order::kColMajor;
  dst_params.rows = bmm_params.lhs_rows;
  dst_params.cols = bmm_params.rhs_cols;

  GemmParams<float, float> gemm_params;
  optimized::Gemm(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, output_data, gemm_params);
}
#endif
} // namespace impl

inline void BatchMatMul(const Shape &lhs_shape, const float *lhs_data, const Shape &rhs_shape,
                        const float *rhs_data, const Shape & /* output_shape */, float *output_data)
{
  const impl::BMMParams bmm_params{lhs_shape, rhs_shape};

#if defined(CKER_X86_PLATFORM)
  impl::bmm_optimized(bmm_params, lhs_data, rhs_data, output_data);
#else
  impl::bmm_reference(bmm_params, lhs_data, rhs_data, output_data);
#endif
}

} // namespace reference
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REFERENCE_BATCH_MATMUL_H__
