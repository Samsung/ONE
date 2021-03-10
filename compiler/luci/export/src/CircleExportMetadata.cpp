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

#include "SerializedData.h"

namespace
{

void write_u32(std::vector<uint8_t> &to, uint32_t value)
{
  to.emplace_back(0xFF & (value >> 0 * 8));
  to.emplace_back(0xFF & (value >> 1 * 8));
  to.emplace_back(0xFF & (value >> 2 * 8));
  to.emplace_back(0xFF & (value >> 3 * 8));
}

} // namespace

namespace luci
{

// 'source_table' is encoded to binary format.
const std::vector<uint8_t> CircleExportMetadata::encoded_source_table(void)
{
  std::vector<uint8_t> data;

  write_u32(data, _source_table.size());

  for (auto &kv : _source_table)
  {
    const auto id = kv.first;
    write_u32(data, id);

    const auto origin_name = kv.second;
    const auto length = origin_name.length();
    write_u32(data, length + 1); // name + '\0

    for (uint32_t i = 0; i < length; ++i)
    {
      data.emplace_back(origin_name.at(i));
    }
    data.emplace_back('\0');
  }

  return data;
}

// 'op_table' is encoded to binary format.
const std::vector<uint8_t> CircleExportMetadata::encoded_op_table(void)
{
  std::vector<uint8_t> data;

  write_u32(data, _op_table.size());

  for (auto &kv : _op_table)
  {
    const auto id = kv.first;
    write_u32(data, id);

    const auto origins = kv.second;
    const auto node_num = origins.size();
    write_u32(data, node_num);

    for (auto origin : origins)
    {
      write_u32(data, origin);
    }
  }

  return data;
}

} // namespace luci
