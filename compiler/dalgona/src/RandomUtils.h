/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __DALGONA_RANDOM_UTILS_H__
#define __DALGONA_RANDOM_UTILS_H__

#include <cstdint>
#include <vector>
#include <random>
#include <stdexcept>

namespace dalgona
{

template <typename T> std::vector<T> genRandomIntData(uint32_t num_elements, T min, T max)
{
  if (min > max)
    throw std::invalid_argument("min is greater than max");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dist(min, max);
  std::vector<T> buffer(num_elements);

  // Write random data
  for (auto &iter : buffer)
    iter = dist(gen);

  return buffer;
}

std::vector<float> genRandomFloatData(uint32_t num_elements, float min, float max);

} // namespace dalgona

#endif // __DALGONA_RANDOM_UTILS_H__
