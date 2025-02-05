/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_OPTIMIZED_GEMM_H__
#define __NNFW_CKER_OPTIMIZED_GEMM_H__

#include "cker/eigen/eigen_gemm_eigen.h"
#include "cker/Shape.h"
#include "cker/Types.h"

#include <ruy/context.h>

namespace nnfw
{
namespace cker
{
namespace optimized
{

#if defined(CKER_X86_PLATFORM)

/* From tensorflow/tensorflow/lite/kernels/cpu_backend_gemm_x86.h */
template <typename LhsScalar, typename RhsScalar, typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct GemmImplX86
{
  static void Run(const MatrixParams<LhsScalar> &, const LhsScalar *,
                  const MatrixParams<RhsScalar> &, const RhsScalar *,
                  const MatrixParams<DstScalar> &, DstScalar *,
                  const GemmParams<AccumScalar, DstScalar, quantization_flavor> &)
  {
    static_assert(
      std::is_floating_point<LhsScalar>::value && std::is_floating_point<RhsScalar>::value &&
        std::is_floating_point<AccumScalar>::value && std::is_floating_point<DstScalar>::value &&
        quantization_flavor != QuantizationFlavor::kFloatingPoint,
      "GemmImplX86 does not supported types other than float yet.");
  }
};

// For float, defer to eigen for now.
template <> struct GemmImplX86<float, float, float, float, QuantizationFlavor::kFloatingPoint>
{
  static void Run(const MatrixParams<float> &lhs_params, const float *lhs_data,
                  const MatrixParams<float> &rhs_params, const float *rhs_data,
                  const MatrixParams<float> &dst_params, float *dst_data,
                  const GemmParams<float, float, QuantizationFlavor::kFloatingPoint> &params)
  {
    detail::GemmImplUsingEigen::Run(lhs_params, lhs_data, rhs_params, rhs_data, dst_params,
                                    dst_data, params);
  }
};

/* From tensorflow/tensorflow/lite/kernels/cpu_backend_gemm.h */
/* GEMM dispatch implementation for x86.
 */
template <typename LhsScalar, typename RhsScalar, typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct GemmImpl : GemmImplX86<LhsScalar, RhsScalar, AccumScalar, DstScalar, quantization_flavor>
{
};

/* From tensorflow/tensorflow/lite/kernels/cpu_backend_gemm.h */
template <typename LhsScalar, typename RhsScalar, typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<LhsScalar> &lhs_params, const LhsScalar *lhs_data,
          const MatrixParams<RhsScalar> &rhs_params, const RhsScalar *rhs_data,
          const MatrixParams<DstScalar> &dst_params, DstScalar *dst_data,
          const GemmParams<AccumScalar, DstScalar, quantization_flavor> &params)
{
  // Generic case: dispatch to any backend as a general GEMM.
  GemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar, quantization_flavor>::Run(
    lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data, params);
}

// From tensorflow/tensorflow/lite/kernels/cpu_backend_gemm_params.h
inline CachePolicy DefaultCachePolicy(bool is_constant_data)
{
  return is_constant_data ? CachePolicy::kCacheIfLargeSpeedup : CachePolicy::kNeverCache;
}
#endif // CKER_X86_PLATFORM

} // namespace optimized
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_GEMM_H__
