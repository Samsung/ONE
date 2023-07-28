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

#include "souschef/Data/Gaussian.h"
#include "souschef/LexicalCast.h"

#include <random>
#include <chrono>

#include <cassert>
#include <stdexcept>
#include <limits> // std::numeric_limits

#include <fp16.h>

namespace souschef
{

template <typename T>
static std::vector<uint8_t> generate_gaussian(int32_t count, float mean, float stddev,
                                              std::minstd_rand::result_type seed)
{
  std::minstd_rand rand{static_cast<std::minstd_rand::result_type>(seed)};
  std::normal_distribution<float> dist{mean, stddev};

  std::vector<uint8_t> res;

  constexpr float max_cap = std::numeric_limits<T>::max();
  constexpr float min_cap = std::numeric_limits<T>::lowest();
  for (uint32_t n = 0; n < count; ++n)
  {
    float raw_value = dist(rand);
    const float capped_value = std::max(min_cap, std::min(max_cap, raw_value));
    auto const value = static_cast<T>(capped_value);
    auto const arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(T); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

template <typename T>
static std::vector<uint8_t> generate_gaussian(int32_t count, float mean, float stddev)
{
  auto time_stamp = std::chrono::system_clock::now().time_since_epoch().count();

  // Note this is implementation defined, change if needed.
  auto seed = static_cast<std::minstd_rand::result_type>(time_stamp);

  return generate_gaussian<T>(count, mean, stddev, seed);
}

std::vector<uint8_t> GaussianFloat32DataChef::generate(int32_t count) const
{
  return generate_gaussian<float>(count, _mean, _stddev);
}

std::vector<uint8_t> GaussianFloat16DataChef::generate(int32_t count) const
{
  auto time_stamp = std::chrono::system_clock::now().time_since_epoch().count();
  auto seed = static_cast<std::minstd_rand::result_type>(time_stamp);

  std::minstd_rand rand{static_cast<std::minstd_rand::result_type>(seed)};
  std::normal_distribution<float> dist{_mean, _stddev};

  std::vector<uint8_t> res;

  constexpr float max_cap = 1e9;
  constexpr float min_cap = -1e9;
  for (uint32_t n = 0; n < count; ++n)
  {
    float raw_value = dist(rand);
    const float capped_value = std::max(min_cap, std::min(max_cap, raw_value));
    const uint16_t value = fp16_ieee_from_fp32_value(capped_value);
    auto const arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(uint16_t); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

std::vector<uint8_t> GaussianInt32DataChef::generate(int32_t count) const
{
  return generate_gaussian<int32_t>(count, _mean, _stddev);
}

std::vector<uint8_t> GaussianInt16DataChef::generate(int32_t count) const
{
  return generate_gaussian<int16_t>(count, _mean, _stddev);
}

std::vector<uint8_t> GaussianUint8DataChef::generate(int32_t count) const
{
  return generate_gaussian<uint8_t>(count, _mean, _stddev);
}

std::vector<uint8_t> GaussianInt8DataChef::generate(int32_t count) const
{
  return generate_gaussian<int8_t>(count, _mean, _stddev);
}

std::unique_ptr<DataChef> GaussianFloat32DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianFloat32DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianInt32DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianInt32DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianInt16DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianInt16DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianUint8DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianUint8DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianInt8DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianInt8DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianFloat16DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianFloat16DataChef{mean, stddev}};
}

} // namespace souschef
