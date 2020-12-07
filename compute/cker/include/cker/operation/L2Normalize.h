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

#ifndef __NNFW_CKER_L2NORMALIZE_H__
#define __NNFW_CKER_L2NORMALIZE_H__

#include "cker/Shape.h"
#include "cker/Utils.h"
#include "cker/Types.h"

namespace nnfw
{
namespace cker
{

void L2NormalizeFloat32(const Shape &input_shape, const float *input_data,
                        const Shape &output_shape, float *output_data)
{
  float epsilon = 1e-6;
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size = MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth = MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  for (int i = 0; i < outer_size; ++i)
  {
    float squared_l2_norm = 0;
    for (int c = 0; c < depth; ++c)
    {
      const float val = input_data[c];
      squared_l2_norm += val * val;
    }
    float l2_norm = std::sqrt(squared_l2_norm);
    l2_norm = std::max(l2_norm, epsilon);
    for (int c = 0; c < depth; ++c)
    {
      *output_data = *input_data / l2_norm;
      ++output_data;
      ++input_data;
    }
  }
}

void L2NormalizeQuant8(L2NormParams &params, const Shape &input_shape, const uint8_t *input_data,
                       const Shape &output_shape, uint8_t *output_data)
{
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth = MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  const int outer_size = MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int32_t input_zero_point = params.input_zero_point;

  for (int i = 0; i < outer_size; ++i)
  {
    int32_t square_l2_norm = 0;
    for (int c = 0; c < depth; c++)
    {
      // Note that input_data advances by depth in the second pass below.
      int32_t diff = input_data[c] - input_zero_point;
      square_l2_norm += diff * diff;
    }
    int32_t inv_l2norm_multiplier;
    int inv_l2norm_shift;
    GetInvSqrtQuantizedMultiplierExp(square_l2_norm, -1, &inv_l2norm_multiplier, &inv_l2norm_shift);
    for (int c = 0; c < depth; c++)
    {
      int32_t diff = *input_data - input_zero_point;
      int32_t rescaled_diff = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        128 * diff, inv_l2norm_multiplier, inv_l2norm_shift);
      int32_t unclamped_output_val = 128 + rescaled_diff;
      int32_t output_val = std::min(static_cast<int32_t>(255),
                                    std::max(static_cast<int32_t>(0), unclamped_output_val));
      *output_data = static_cast<uint8_t>(output_val);
      ++input_data;
      ++output_data;
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_L2NORMALIZE_H__
