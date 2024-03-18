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

#ifndef __SOUSCHEF_DATA_EXPLICIT_H__
#define __SOUSCHEF_DATA_EXPLICIT_H__

#include "souschef/DataChef.h"
#include "souschef/LexicalCast.h"

#include <vector>

namespace souschef
{

template <typename T> class ExplicitDataChef final : public DataChef
{
public:
  ExplicitDataChef()
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override
  {
    std::vector<uint8_t> res;

    for (uint32_t n = 0; n < count; ++n)
    {
      T const value = (n < _values.size()) ? _values.at(n) : T{};
      const uint8_t *arr = reinterpret_cast<const uint8_t *>(&value);

      for (uint32_t b = 0; b < sizeof(T); ++b)
      {
        res.emplace_back(arr[b]);
      }
    }

    return res;
  }

public:
  void insert(const T &value) { _values.emplace_back(value); }

private:
  std::vector<T> _values;
};

template <> class ExplicitDataChef<std::string> final : public DataChef
{
public:
  ExplicitDataChef()
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override;

public:
  void insert(const std::string &value) { _values.emplace_back(value); }

private:
  void write_value(std::vector<uint8_t> &res, int32_t value) const;

private:
  std::vector<std::string> _values;
};

template <typename T> struct ExplicitDataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const
  {
    std::unique_ptr<ExplicitDataChef<T>> res{new ExplicitDataChef<T>};

    for (uint32_t n = 0; n < args.count(); ++n)
    {
      auto const value = to_number<T>(args.value(n));
      res->insert(value);
    }

    return std::move(res);
  }
};

class ExplicitFloat16DataChef final : public DataChef
{
public:
  ExplicitFloat16DataChef()
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override;

public:
  void insert(const float &value) { _values.emplace_back(value); }

private:
  // NOTE store values in float but will convert to uint16_t in generate()
  std::vector<float> _values;
};

struct ExplicitFloat16DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const
  {
    std::unique_ptr<ExplicitFloat16DataChef> res{new ExplicitFloat16DataChef};

    for (uint32_t n = 0; n < args.count(); ++n)
    {
      auto const value = to_number<float>(args.value(n));
      res->insert(value);
    }

    return std::move(res);
  }
};

class ExplicitInt4DataChef final : public DataChef
{
public:
  ExplicitInt4DataChef()
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override;

public:
  void insert(const int8_t &value) { _values.emplace_back(value); }

private:
  std::vector<int8_t> _values;
};

struct ExplicitInt4DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const
  {
    std::unique_ptr<ExplicitInt4DataChef> res{new ExplicitInt4DataChef};

    for (uint32_t n = 0; n < args.count(); ++n)
    {
      auto const data = to_number<int8_t>(args.value(n));
      res->insert(data);
    }

    return std::move(res);
  }
};

class ExplicitUint4DataChef final : public DataChef
{
public:
  ExplicitUint4DataChef()
  {
    // DO NOTHING
  }

public:
  std::vector<uint8_t> generate(int32_t count) const override;

public:
  void insert(const uint8_t &value) { _values.emplace_back(value); }

private:
  std::vector<uint8_t> _values;
};

struct ExplicitUint4DataChefFactory : public DataChefFactory
{
  std::unique_ptr<DataChef> create(const Arguments &args) const
  {
    std::unique_ptr<ExplicitUint4DataChef> res{new ExplicitUint4DataChef};

    for (uint32_t n = 0; n < args.count(); ++n)
    {
      auto const data = to_number<uint8_t>(args.value(n));
      res->insert(data);
    }

    return std::move(res);
  }
};

} // namespace souschef

#endif // __SOUSCHEF_DATA_EXPLICIT_H__
