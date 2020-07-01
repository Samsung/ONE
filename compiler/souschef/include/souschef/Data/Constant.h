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

#ifndef __SOURCHEF_DATA_CONSTANT_H__
#define __SOURCHEF_DATA_CONSTANT_H__

#include "souschef/DataChef.h"
#include "souschef/LexicalCast.h"

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

} // namespace souschef

#endif // __SOURCHEF_DATA_CONSTANT_H__
