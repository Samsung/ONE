/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Gaussian.h"
#include "LexicalCast.h"

#include <random>
#include <chrono>

#include <cassert>
#include <stdexcept>

std::vector<uint8_t> GaussianFloat32DataChef::generate(int32_t count) const
{
  // TODO Support seed value override
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();

  std::minstd_rand rand{static_cast<std::minstd_rand::result_type>(seed)};
  std::normal_distribution<float> dist{_mean, _stddev};

  std::vector<uint8_t> res;

  for (uint32_t n = 0; n < count; ++n)
  {
    auto const value = dist(rand);
    auto const arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(float); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

std::vector<uint8_t> GaussianInt32DataChef::generate(int32_t count) const
{
  // TODO Support seed value override
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();

  std::minstd_rand rand{static_cast<std::minstd_rand::result_type>(seed)};
  std::normal_distribution<float> dist{_mean, _stddev};

  std::vector<uint8_t> res;

  for (uint32_t n = 0; n < count; ++n)
  {
    auto const value = static_cast<int32_t>(dist(rand));
    auto const arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(int32_t); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

std::vector<uint8_t> GaussianUint8DataChef::generate(int32_t count) const
{
  // TODO Support seed value override
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();

  std::minstd_rand rand{static_cast<std::minstd_rand::result_type>(seed)};
  std::normal_distribution<float> dist{_mean, _stddev};

  std::vector<uint8_t> res;

  for (uint32_t n = 0; n < count; ++n)
  {
    auto const value = static_cast<uint8_t>(dist(rand));        // uint8_t for data type
    auto const arr = reinterpret_cast<const uint8_t *>(&value); // uint8_t for byte streaming

    for (uint32_t b = 0; b < sizeof(uint8_t); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
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
