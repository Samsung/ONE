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
#include "cker/operation/Helper/BatchMatMulParams.h"

namespace nnfw
{
namespace cker
{
namespace reference
{

inline void BatchMatMul(const BatchMatMulParams &params, const float *lhs_data,
                        const float *rhs_data, float *output_data)
{
  for (int b0 = 0; b0 < params.batch_dim0; ++b0)
  {
    const float *lhs_ptr0 = lhs_data + (b0 * params.lhs_ext0);
    const float *rhs_ptr0 = rhs_data + (b0 * params.rhs_ext0);
    for (int b1 = 0; b1 < params.batch_dim1; ++b1)
    {
      const float *lhs_ptr1 = lhs_ptr0 + b1 * params.lhs_ext1;
      const float *rhs_ptr1 = rhs_ptr0 + b1 * params.rhs_ext1;
      for (int b2 = 0; b2 < params.batch_dim2; ++b2)
      {
        const float *lhs_ptr2 = lhs_ptr1 + b2 * params.lhs_ext2;
        const float *rhs_ptr2 = rhs_ptr1 + b2 * params.rhs_ext2;
        float *out_ptr = output_data + ((b0 * params.batch_dim1 * params.batch_dim2) +
                                        b1 * params.batch_dim2 + b2) *
                                         params.lhs_rows * params.rhs_cols;
        for (int j = 0; j < params.rhs_cols; ++j)
        {
          for (int i = 0; i < params.lhs_rows; ++i)
          {
            float total = 0.f;
            for (int k = 0; k < params.accum_depth; ++k)
            {
              total += lhs_ptr2[params.accum_depth * i + k] * rhs_ptr2[j * params.accum_depth + k];
            }
            int idx = params.lhs_rows * j + i;
            out_ptr[idx] = total;
          }
        }
      }
    }
  }
}

} // namespace reference
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_REFERENCE_BATCH_MATMUL_H__
