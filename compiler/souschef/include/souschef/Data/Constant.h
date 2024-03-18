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

#ifndef __SOUSCHEF_DATA_CONSTANT_H__
#define __SOUSCHEF_DATA_CONSTANT_H__

#include "souschef/DataChef.h"
#include "souschef/LexicalCast.h"

#include <stdexcept>

namespace souschef
{

template <typename T> class ConstantDataChef final : public DataChef
{
public:
  ConstantDataChef(const T &value) : _value{value}
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override
  {
    std::vector<uint8_t> res;

    for (uint32_t n = 0; n < count; ++n)
    {
      const uint8_t *arr = reinterpret_cast<const uint8_t *>(&_value);

      for (uint32_t b = 0; b < sizeof(T); ++b)
      {
        res.emplace_back(arr[b]);
      }
    }

    return res;
  }

private:
  T _value;
};

template <typename T> struct ConstantDataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const
  {
    auto const value = to_number<T>(args.value(0));
    return std::unique_ptr<DataChef>{new ConstantDataChef<T>{value}};
  }
};

class ConstantInt4DataChef final : public DataChef
{
public:
  ConstantInt4DataChef(const int8_t &value) : _value{value}
  {
    // DO NOTHING
  }

public:
  // int4 constant is saved as int8 (extra 4 bits are filled with sign bits).
  // Callers must cast each element to int8 before using it.
  // Example)
  //   ConstInt4DataChef chef(-5)
  //   auto values = chef.generate(3);
  //   for (uint8_t value: values) {
  //     int8_t real_value = static_cast<int8_t>(value);
  //     assert(value == 251 and real_value == -5);
  std::vector<uint8_t> generate(int32_t count) const override
  {
    std::vector<uint8_t> res;

    if (_value < -8 || 7 < _value)
      throw std::runtime_error("Constant value out of range.");

    for (uint32_t n = 0; n < count; ++n)
    {
      const uint8_t data = static_cast<const uint8_t>(_value);
      res.emplace_back(data);
    }

    return res;
  }

private:
  int8_t _value;
};

struct ConstantInt4DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const
  {
    auto const value = to_number<int8_t>(args.value(0));
    return std::unique_ptr<DataChef>{new ConstantInt4DataChef{value}};
  }
};

class ConstantUint4DataChef final : public DataChef
{
public:
  ConstantUint4DataChef(const uint8_t &value) : _value{value}
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override
  {
    std::vector<uint8_t> res;

    if (15 < _value)
      throw std::runtime_error("Constant value out of range.");

    for (uint32_t n = 0; n < count; ++n)
    {
      res.emplace_back(_value);
    }

    return res;
  }

private:
  uint8_t _value;
};

struct ConstantUint4DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const
  {
    auto const value = to_number<uint8_t>(args.value(0));
    return std::unique_ptr<DataChef>{new ConstantUint4DataChef{value}};
  }
};

} // namespace souschef

#endif // __SOUSCHEF_DATA_CONSTANT_H__
