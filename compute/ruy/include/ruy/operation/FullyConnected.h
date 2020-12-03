/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_RUY_FULLY_CONNECTED_H__
#define __NNFW_RUY_FULLY_CONNECTED_H__

#include "ruy/Shape.h"
#include "ruy/Types.h"
#include "ruy/Utils.h"
#include "ruy/RuySupport.h"

#include <ruy/ruy.h>
#include <ruy/context.h>

namespace nnfw
{
namespace ruy
{

inline void FullyConnected(const FullyConnectedParams &params, const Shape &input_shape,
                           const float *input_data, const Shape &weights_shape,
                           const float *weights_data, const Shape &,
                           const float *optional_bias_data, const Shape &output_shape,
                           float *output_data, ::ruy::Context *ruy_context)
{
  const int dims_count = weights_shape.DimensionsCount();
  const int input_rows = weights_shape.Dims(dims_count - 1);
  MatrixParams<float> rhs_params;
  rhs_params.order = Order::kColMajor;
  rhs_params.rows = input_rows;
  rhs_params.cols = input_shape.FlatSize() / input_rows;
  rhs_params.cache_policy = DefaultCachePolicy(params.rhs_cacheable);
  assert(input_shape.FlatSize() == (rhs_params.rows * rhs_params.cols));
  MatrixParams<float> lhs_params;
  lhs_params.order = Order::kRowMajor;
  lhs_params.cols = weights_shape.Dims(dims_count - 1);
  lhs_params.rows = FlatSizeSkipDim(weights_shape, dims_count - 1);
  lhs_params.cache_policy = DefaultCachePolicy(params.lhs_cacheable);
  MatrixParams<float> dst_params;
  dst_params.order = Order::kColMajor;
  dst_params.rows = output_shape.Dims(output_shape.DimensionsCount() - 1);
  dst_params.cols = FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  GemmParams<float, float> gemm_params;
  gemm_params.bias = optional_bias_data;
  gemm_params.clamp_min = params.float_activation_min;
  gemm_params.clamp_max = params.float_activation_max;

  // Below code was copied from tflite::cpu_backend_gemm::detail::GemmImplUsingRuy
  ::ruy::Matrix<float> ruy_lhs;
  ::ruy::Matrix<float> ruy_rhs;
  ::ruy::Matrix<float> ruy_dst;
  // Note that cache is always enabled for input and weight tensors
  ruy_support::MakeRuyMatrix(lhs_params, weights_data, &ruy_lhs, true);
  ruy_support::MakeRuyMatrix(rhs_params, input_data, &ruy_rhs, true);
  ruy_support::MakeRuyMatrix(dst_params, output_data, &ruy_dst);

  ::ruy::BasicSpec<float, float> ruy_mul_params;
  ruy_support::MakeRuyMulParams(gemm_params, &ruy_mul_params);

  ::ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, ruy_context, &ruy_dst);
}

} // namespace ruy
} // namespace nnfw

#endif // __NNFW_RUY_FULLY_CONNECTED_H__
