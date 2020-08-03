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

#ifndef __NNFW_CKER_RUY_RUY_SUPPORT_H__
#define __NNFW_CKER_RUY_RUY_SUPPORT_H__

#include <util/ConfigSource.h>
#include <ruy/context.h>
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{
namespace ruy_support
{

template <typename Scalar, typename DataPointer>
void MakeRuyMatrix(const MatrixParams<Scalar> &params, DataPointer data_ptr,
                   ruy::Matrix<Scalar> *dst)
{
  dst->layout.rows = params.rows;
  dst->layout.cols = params.cols;
  if (params.order == Order::kColMajor)
  {
    dst->layout.order = ruy::Order::kColMajor;
    dst->layout.stride = params.rows;
  }
  else
  {
    dst->layout.order = ruy::Order::kRowMajor;
    dst->layout.stride = params.cols;
  }
  // Note that ruy::Matrix::data is a ConstCheckingPtr, not a plain pointer.
  // It does care whether we assign to it a Scalar* or a const Scalar*.
  dst->data = data_ptr;
  dst->zero_point = params.zero_point;
  dst->cacheable = params.cacheable;
}

template <typename GemmParamsType, typename RuySpecType>
void MakeRuySpec(const GemmParamsType &params, RuySpecType *ruy_spec)
{
  // This validation has already been performed by the Gemm API entry point,
  // but it doesn't hurt to test specifically this again here, where it's
  // being used.
  ValidateGemmParams(params);

  ruy_spec->multiplier_fixedpoint = params.multiplier_fixedpoint;
  ruy_spec->multiplier_exponent = params.multiplier_exponent;
  ruy_spec->multiplier_fixedpoint_perchannel = params.multiplier_fixedpoint_perchannel;
  ruy_spec->multiplier_exponent_perchannel = params.multiplier_exponent_perchannel;
  ruy_spec->bias = params.bias;
  ruy_spec->clamp_min = params.clamp_min;
  ruy_spec->clamp_max = params.clamp_max;
}

} // namespace ruy_support
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_RUY_RUY_SUPPORT_H__
