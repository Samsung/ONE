/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "benchmark/RandomGenerator.h"

namespace benchmark
{

template <> int8_t RandomGenerator::generate<int8_t>(void)
{
  // The value of type_range is 255.
  float type_range = static_cast<float>(std::numeric_limits<int8_t>::max()) -
                     static_cast<float>(std::numeric_limits<int8_t>::min());
  // Most _dist values range from -5.0 to 5.0.
  float min_range = -5.0f;
  float max_range = 5.0f;
  // NOTE shifted_relative_val has Gaussian distribution that origin mean was 0 and standard
  // deviation was 2. And then its values are distributed and shift to that mean is 127.5 and range
  // is about [0, 255].
  float shifted_relative_val = (_dist(_rand) - min_range) * type_range / (max_range - min_range);

  // shifted_relative_val is adjusted to be mapped to end points of the range, if it is out of range
  // values.
  if (shifted_relative_val < -128.0f)
  {
    return -128;
  }
  else if (shifted_relative_val > type_range)
  {
    return 127;
  }

  // Convert shifted_relative_val from float to int8
  return static_cast<int8_t>(shifted_relative_val);
}

template <> uint8_t RandomGenerator::generate<uint8_t>(void)
{
  // The value of type_range is 255.
  float type_range = static_cast<float>(std::numeric_limits<uint8_t>::max()) -
                     static_cast<float>(std::numeric_limits<uint8_t>::min());
  // Most _dist values range from -5.0 to 5.0.
  float min_range = -5.0f;
  float max_range = 5.0f;
  // NOTE shifted_relative_val has Gaussian distribution that origin mean was 0 and standard
  // deviation was 2. And then its values are distributed and shift to that mean is 127.5 and range
  // is about [0, 255].
  float shifted_relative_val = (_dist(_rand) - min_range) * type_range / (max_range - min_range);

  // shifted_relative_val is adjusted to be mapped to end points of the range, if it is out of range
  // values.
  if (shifted_relative_val < 0.0f)
  {
    return 0;
  }
  else if (shifted_relative_val > type_range)
  {
    return 255;
  }

  // Convert shifted_relative_val from float to uint8
  return static_cast<uint8_t>(shifted_relative_val);
}

template <> bool RandomGenerator::generate<bool>(void)
{
  std::uniform_int_distribution<> dist(0, 1); // [0, 1]
  return dist(_rand);
}

template <> int32_t RandomGenerator::generate<int32_t>(void)
{
  // Instead of INT_MAX, 99 is chosen because int32_t input does not mean
  // that the model can have any value in int32_t can hold.
  // For example, one_hot operation gets indices as int32_t tensor.
  // However, we usually expect it would hold a value in [0..depth).
  // In our given model, depth was 10137.
  const int int32_random_max = 99;
  std::uniform_int_distribution<> dist(0, int32_random_max);
  return dist(_rand);
}

template <> int64_t RandomGenerator::generate<int64_t>(void)
{
  // Instead of INT_MAX, 99 is chosen because int64_t input does not mean
  // that the model can have any value in int64_t can hold.
  // For example, one_hot operation gets indices as int64_t tensor.
  // However, we usually expect it would hold a value in [0..depth).
  // In our given model, depth was 10137.
  const int64_t int64_random_max = 99;
  std::uniform_int_distribution<> dist(0, int64_random_max);
  return dist(_rand);
}

} // namespace benchmark
