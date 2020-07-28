/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_QUANTIZE_H__
#define __NNFW_CKER_QUANTIZE_H__

#include "cker/Shape.h"
#include "cker/Types.h"
#include "cker/Utils.h"
#include <stdexcept>
#include <iostream>
namespace nnfw
{
namespace cker
{
template <typename InputT, typename OutputT>
inline void Quantize(const Shape &input_shape, const InputT *input_data, const Shape &output_shape,
                     OutputT *output_data, const float output_scale, const int32_t output_offset)
{
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  int min_val = std::numeric_limits<OutputT>::min();
  int max_val = std::numeric_limits<OutputT>::max();

  for (int i = 0; i < flat_size; i++)
  {
    int32_t unclamped = static_cast<int32_t>(round(input_data[i] / output_scale)) + output_offset;
    int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_QUANTIZE_H__
