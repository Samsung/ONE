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

#ifndef __NNFW_CKER_RMS_NORM_H__
#define __NNFW_CKER_RMS_NORM_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"

#include <cmath>

namespace nnfw
{
namespace cker
{

inline void RmsNorm(const RmsNormParams &params, const Shape &input_shape, const float *input_data,
                    const Shape &gamma_shape, const float *gamma_data, const Shape &output_shape,
                    float *output_data)
{
  const int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t heights = MatchingDim(input_shape, 1, output_shape, 1);
  const int32_t widths = MatchingDim(input_shape, 2, output_shape, 2);
  const int32_t channels = MatchingDim(input_shape, 3, output_shape, 3);

  if (gamma_shape.DimensionsCount() != 1 ||
      gamma_shape.Dims(0) != input_shape.Dims(input_shape.DimensionsCount() - 1))
    throw std::runtime_error("cker::RmsNorm: Unmatched gamma shape");

  for (int32_t batch = 0; batch < batches; batch++)
  {
    for (int32_t height = 0; height < heights; height++)
    {
      for (int32_t width = 0; width < widths; width++)
      {
        double square_sum = 0.0f;
        for (int32_t channel = 0; channel < channels; channel++)
        {
          double input_val = input_data[Offset(input_shape, batch, height, width, channel)];
          square_sum += (input_val * input_val);
        }
        double rms = std::sqrt((square_sum / channels) + params.epsilon);
        for (int32_t channel = 0; channel < channels; channel++)
        {
          double gamma = (gamma_data ? gamma_data[channel] : 1.0);
          output_data[Offset(output_shape, batch, height, width, channel)] =
            gamma * (input_data[Offset(input_shape, batch, height, width, channel)] / rms);
        }
      }
    }
  }
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_RMS_NORM_H__
