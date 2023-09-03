/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "RecordFunction.h"

#include <luci/IR/CircleQuantParam.h>

#include <cassert>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace record_minmax
{

float getNthPercentile(std::vector<float> &vector, float percentile)
{
  if (percentile < 0 || percentile > 100)
    throw std::runtime_error("Percentile must be ranged from 0 to 100");

  if (vector.empty())
    throw std::runtime_error("Percentile must take a non-empty vector as an argument");

  if (vector.size() == 1)
    return vector[0];

  std::vector<float> copy;
  copy.assign(vector.begin(), vector.end());
  std::sort(copy.begin(), copy.end());

  if (percentile == 0.0)
    return copy.front();

  if (percentile == 100.0)
    return copy.back();

  int index = static_cast<int>(std::floor((copy.size() - 1) * percentile / 100.0));

  float percent_i = static_cast<float>(index) / static_cast<float>(copy.size() - 1);
  float fraction =
    (percentile / 100.0 - percent_i) / ((index + 1.0) / (copy.size() - 1.0) - percent_i);
  float res = copy[index] + fraction * (copy[index + 1] - copy[index]);
  return res;
}

float getMovingAverage(const std::vector<float> &vector, const float alpha,
                       const uint8_t batch_size, bool is_min)
{
  assert(!vector.empty());
  assert(alpha >= 0.0 && alpha <= 1.0);
  assert(batch_size > 0);

  auto getBatchMinOrMax = [&](uint32_t start_index) {
    assert(start_index < vector.size());

    float res = is_min ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();
    for (uint32_t offset = 0; offset < batch_size; offset++)
    {
      uint32_t index = start_index + offset;
      if (index >= vector.size())
        break;

      if (is_min)
      {
        res = vector[index] < res ? vector[index] : res;
      }
      else
      {
        res = vector[index] > res ? vector[index] : res;
      }
    }
    return res;
  };

  float curr_avg = getBatchMinOrMax(0);
  for (uint32_t i = batch_size; i < vector.size(); i += batch_size)
  {
    curr_avg = curr_avg * alpha + getBatchMinOrMax(i) * (1.0 - alpha);
  }
  return curr_avg;
}

} // namespace record_minmax
