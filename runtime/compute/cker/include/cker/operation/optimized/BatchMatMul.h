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

#ifndef __NNFW_CKER_OPTIMIZED_BATCH_MATMUL_H__
#define __NNFW_CKER_OPTIMIZED_BATCH_MATMUL_H__

#include "cker/Shape.h"
#include "cker/operation/Helper/BatchMatMulParams.h"
#include "cker/operation/optimized/Gemm.h"

namespace nnfw
{
namespace cker
{
namespace optimized
{
#if defined(CKER_X86_PLATFORM)

inline void BatchMatMul(const BatchMatMulParams &params, const float *lhs_data,
                        const float *rhs_data, float *output_data)
{
  MatrixParams<float> lhs_params;
  lhs_params.order = Order::kRowMajor; // ignored by GemmImplUsingEigen
  lhs_params.rows = params.lhs_rows;
  lhs_params.cols = params.lhs_cols;

  MatrixParams<float> rhs_params;
  lhs_params.order = Order::kRowMajor; // ignored by GemmImplUsingEigen
  rhs_params.rows = params.rhs_rows;
  rhs_params.cols = params.rhs_cols;

  MatrixParams<float> dst_params;
  lhs_params.order = Order::kRowMajor; // ignored by GemmImplUsingEigen
  dst_params.rows = params.lhs_rows;
  dst_params.cols = params.rhs_cols;

  for (int b0 = 0; b0 < params.batch_dim0; ++b0)
  {
    for (int b1 = 0; b1 < params.batch_dim1; ++b1)
    {
      for (int b2 = 0; b2 < params.batch_dim2; ++b2)
      {
        const float *lhs_ptr =
          lhs_data + b0 * params.lhs_ext0 + b1 * params.lhs_ext1 + b2 * params.lhs_ext2;
        const float *rhs_ptr =
          rhs_data + b0 * params.rhs_ext0 + b1 * params.rhs_ext1 + b2 * params.rhs_ext2;
        float *out_ptr = output_data + ((b0 * params.batch_dim1 * params.batch_dim2) +
                                        b1 * params.batch_dim2 + b2) *
                                         params.lhs_rows * params.rhs_cols;

        optimized::Gemm(lhs_params, lhs_ptr, rhs_params, rhs_ptr, dst_params, out_ptr,
                        GemmParams<float, float>{});
      }
    }
  }
}
#endif
} // namespace optimized
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_BATCH_MATMUL_H__
