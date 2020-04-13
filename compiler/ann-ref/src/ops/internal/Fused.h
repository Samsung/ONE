/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __FUSED_H__
#define __FUSED_H__

#include "Dims.h"

#include <cstdint>

enum class FusedActivationFunc : int32_t {
  NONE = 0,
  RELU = 1,
  RELU1 = 2,
  RELU6 = 3,
};

enum class FusedActivationFunctionType
{
  kNone,
  kRelu6,
  kRelu1,
  kRelu
};

template <FusedActivationFunctionType Ac> struct ActivationFunctionImpl;

template <> struct ActivationFunctionImpl<FusedActivationFunctionType::kNone>
{
  static float Eval(float x) { return x; }
};

template <> struct ActivationFunctionImpl<FusedActivationFunctionType::kRelu>
{
  static float Eval(float x) { return x < 0.f ? 0.f : x; }
};

template <> struct ActivationFunctionImpl<FusedActivationFunctionType::kRelu1>
{
  static float Eval(float x) { return x > 1.f ? 1.f : x < -1.f ? -1.f : x; }
};

template <> struct ActivationFunctionImpl<FusedActivationFunctionType::kRelu6>
{
  static float Eval(float x) { return x > 6.f ? 6.f : x < 0.f ? 0.f : x; }
};

template <FusedActivationFunctionType Ac> float ActivationFunction(float x)
{
  return ActivationFunctionImpl<Ac>::Eval(x);
}

template <FusedActivationFunctionType Ac>
void AddBiasAndEvalActivationFunction(const float *bias_data, const Dims<4> &bias_dims,
                                      float *array_data, const Dims<4> &array_dims)
{
  const int bias_size = bias_dims.sizes[3] * bias_dims.strides[3];
  const int array_size = array_dims.sizes[3] * array_dims.strides[3];
  DCHECK_EQ((array_size % bias_size), 0);
  for (int array_offset = 0; array_offset < array_size; array_offset += bias_size)
  {
    for (int i = 0; i < bias_size; i++)
    {
      array_data[array_offset + i] =
          ActivationFunction<Ac>(array_data[array_offset + i] + bias_data[i]);
    }
  }
}

#endif // __FUSED_H__
