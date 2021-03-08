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

#include "ExportMetadata.h"

namespace luci
{

void ExportMetadata::link_source(uint32_t node_id, uint32_t source_id)
{
  if (_op_table.find(node_id) == _op_table.end())
    _op_table.emplace(node_id, std::set<uint32_t>());
  _op_table.at(node_id).emplace(source_id);
}

void ExportMetadata::add_source(uint32_t source_id, std::string origin_name)
{
  if (_source_table.find(source_id) == _source_table.end())
    _source_table.emplace(source_id, origin_name);
  else
  {
    // NOTE Model with only one subgraph must not reach here.
    //      However, model with multiple subgraph can reach here.
    //      For now, as we do not consider about multiple subgraph in profiling,
    //      just do nothing and support those cases in the future.
  }
}

} // namespace luci

namespace luci
{

const std::vector<uint8_t> encoded_source_table(ExportMetadata &metadata)
{
  std::vector<uint8_t> data;

  const auto source_table = metadata.source_table();

  const auto size = source_table.size();
  data.emplace_back(0xFF & (size >> 0 * 8));
  data.emplace_back(0xFF & (size >> 1 * 8));
  data.emplace_back(0xFF & (size >> 2 * 8));
  data.emplace_back(0xFF & (size >> 3 * 8));
  for (auto &kv : source_table)
  {
    const auto id = kv.first;
    data.emplace_back(0xFF & (id >> 0 * 8));
    data.emplace_back(0xFF & (id >> 1 * 8));
    data.emplace_back(0xFF & (id >> 2 * 8));
    data.emplace_back(0xFF & (id >> 3 * 8));

    const auto origin_name = kv.second;
    const auto length = origin_name.length() + 1; // name + '\0'
    data.emplace_back(0xFF & (length >> 0 * 8));
    data.emplace_back(0xFF & (length >> 1 * 8));
    data.emplace_back(0xFF & (length >> 2 * 8));
    data.emplace_back(0xFF & (length >> 3 * 8));

    for (uint32_t i = 0; i < length - 1; ++i)
    {
      data.emplace_back(origin_name.at(i));
    }
    data.emplace_back(0);
  }

  return data;
}

const std::vector<uint8_t> encoded_op_table(ExportMetadata &metadata)
{
  std::vector<uint8_t> data;

  const auto op_table = metadata.op_table();

  const auto size = op_table.size();
  data.emplace_back(0xFF & (size >> 0 * 8));
  data.emplace_back(0xFF & (size >> 1 * 8));
  data.emplace_back(0xFF & (size >> 2 * 8));
  data.emplace_back(0xFF & (size >> 3 * 8));
  for (auto &kv : op_table)
  {
    const auto id = kv.first;
    data.emplace_back(0xFF & (id >> 0 * 8));
    data.emplace_back(0xFF & (id >> 1 * 8));
    data.emplace_back(0xFF & (id >> 2 * 8));
    data.emplace_back(0xFF & (id >> 3 * 8));

    const auto origins = kv.second;
    const auto node_num = origins.size();
    data.emplace_back(0xFF & (node_num >> 0 * 8));
    data.emplace_back(0xFF & (node_num >> 1 * 8));
    data.emplace_back(0xFF & (node_num >> 2 * 8));
    data.emplace_back(0xFF & (node_num >> 3 * 8));

    for (auto origin : origins)
    {
      data.emplace_back(0xFF & (origin >> 0 * 8));
      data.emplace_back(0xFF & (origin >> 1 * 8));
      data.emplace_back(0xFF & (origin >> 2 * 8));
      data.emplace_back(0xFF & (origin >> 3 * 8));
    }
  }

  return data;
}

} // namespace luci
