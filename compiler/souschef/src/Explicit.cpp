/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "souschef/Data/Explicit.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <fp16.h>

namespace souschef
{

/**
 * @note This emulates TensorFlow int DynamicBuffer::WriteToBuffer(char** buffer) method
 *       Memory structure:
 *         int32_t count
 *         int32_t offsets[count + 1]
 *         string values[count]
 *       where string is like std::string without ending null byte
 */
std::vector<uint8_t> ExplicitDataChef<std::string>::generate(int32_t count) const
{
  std::vector<uint8_t> res;

  // write count
  write_value(res, count);

  // write first item offset
  int32_t start = sizeof(int32_t) * (count + 2);
  write_value(res, start);

  // write succeeding items offset (or the end)
  int32_t offset = start;
  for (uint32_t n = 0; n < count; ++n)
  {
    std::string const value = (n < _values.size()) ? _values.at(n) : std::string{};
    offset += value.length();
    write_value(res, offset);
  }

  for (uint32_t n = 0; n < count; ++n)
  {
    std::string const value = (n < _values.size()) ? _values.at(n) : std::string{};
    const uint8_t *arr = reinterpret_cast<const uint8_t *>(value.c_str());

    for (uint32_t b = 0; b < value.length(); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

void ExplicitDataChef<std::string>::write_value(std::vector<uint8_t> &res, int32_t value) const
{
  const uint8_t *arr = reinterpret_cast<const uint8_t *>(&value);

  for (uint32_t b = 0; b < sizeof(int32_t); ++b)
  {
    res.emplace_back(arr[b]);
  }
}

std::vector<uint8_t> ExplicitFloat16DataChef::generate(int32_t count) const
{
  std::vector<uint8_t> res;

  for (uint32_t n = 0; n < count; ++n)
  {
    float const fvalue = (n < _values.size()) ? _values.at(n) : 0.0;
    uint16_t const value = fp16_ieee_from_fp32_value(fvalue);
    auto const arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(uint16_t); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

std::vector<uint8_t> ExplicitInt4DataChef::generate(int32_t count) const
{
  std::vector<uint8_t> res;

  for (uint32_t n = 0; n < count; ++n)
  {
    int8_t const value = (n < _values.size()) ? _values.at(n) : 0;
    if (value < -8 || 7 < value)
      throw std::runtime_error("Explicit value out of range.");

    const uint8_t data = static_cast<const uint8_t>(value);
    res.emplace_back(data);
  }

  return res;
}

std::vector<uint8_t> ExplicitUint4DataChef::generate(int32_t count) const
{
  std::vector<uint8_t> res;

  for (uint32_t n = 0; n < count; ++n)
  {
    uint8_t const value = (n < _values.size()) ? _values.at(n) : 0;
    if (15 < value)
      throw std::runtime_error("Explicit value out of range.");

    res.emplace_back(value);
  }

  return res;
}

} // namespace souschef
