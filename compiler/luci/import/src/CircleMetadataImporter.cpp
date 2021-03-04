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

#include "CircleMetadataImporter.h"

#include <set>
#include <vector>

namespace
{

uint32_t read_u32(const std::vector<uint8_t> &buffer, uint32_t idx)
{
  uint32_t val = 0;
  val += (buffer.at(idx + 0) << 0 * 8);
  val += (buffer.at(idx + 1) << 1 * 8);
  val += (buffer.at(idx + 2) << 2 * 8);
  val += (buffer.at(idx + 3) << 3 * 8);
  return val;
}

} // namespace

namespace
{

/**
 * 'source_table'
 * - [ entry_number : uint32_t ]
 * - [ id : uint32_t ][ length : uint32_t ][ data : 'length' Bytes ] * entry_number
 */
const std::map<uint32_t, std::string>
decoded_source_table(const std::vector<uint8_t> &source_table_data)
{
  std::map<uint32_t, std::string> source_id_name_map;
  uint32_t idx = 0;

  uint32_t table_size = read_u32(source_table_data, idx);
  idx += 4;

  while (idx < source_table_data.size())
  {
    uint32_t id = read_u32(source_table_data, idx);
    idx += 4;

    uint32_t length = read_u32(source_table_data, idx);
    idx += 4;

    std::string origin_name;
    for (uint32_t j = 0; j < length; ++j)
      origin_name += (char)source_table_data.at(idx + j);
    idx += length;

    if (source_id_name_map.insert({id, origin_name}).second == false)
      throw std::runtime_error("Duplicated origin ID");
  }
  assert(idx == source_table_data.size());
  assert(source_id_name_map.size() == table_size);

  return source_id_name_map;
}

/**
 * 'op_table'
 * - [ entry_number : uitn32_t ]
 * - [ id : uint32_t ][ node_num : uint32_t ][ node_ids : node_num * uint32_t ] * entry_number
 */
const std::map<uint32_t, std::set<uint32_t>>
decoded_op_table(const std::vector<uint8_t> &op_table_data)
{
  std::map<uint32_t, std::set<uint32_t>> node_source_ids_map;
  uint32_t idx = 0;

  uint32_t table_size = read_u32(op_table_data, idx);
  idx += 4;

  while (idx < op_table_data.size())
  {
    uint32_t id = read_u32(op_table_data, idx);
    idx += 4;

    uint32_t node_num = read_u32(op_table_data, idx);
    idx += 4;

    std::set<uint32_t> source_ids;
    for (uint32_t j = 0; j < node_num; ++j)
    {
      uint32_t origin = read_u32(op_table_data, idx);
      idx += 4;

      source_ids.insert(origin);
    }

    if (node_source_ids_map.insert({id, source_ids}).second == false)
      throw std::runtime_error("Duplicated origin ID");
  }
  assert(idx == op_table_data.size());
  assert(node_source_ids_map.size() == table_size);

  return node_source_ids_map;
}

} // namespace

namespace luci
{

CircleImportMetadata::CircleImportMetadata(const luci::CircleReader &reader)
{
  const auto &metadata = reader.metadata();
  for (uint32_t i = 0; i < metadata.size(); ++i)
  {
    const circle::MetadataT &meta = *metadata[i];

    assert(meta.buffer < reader.buffers().size());
    const std::vector<uint8_t> &buffer = reader.buffers()[meta.buffer]->data;

    if (meta.name.compare("ONE_op_table") == 0)
      _op_table = decoded_op_table(buffer);
    else if (meta.name.compare("ONE_source_table") == 0)
      _source_table = decoded_source_table(buffer);
  }
}

const std::map<uint32_t, std::shared_ptr<CircleNodeOrigin>> CircleImportMetadata::origin_table(void)
{
  std::map<uint32_t, std::shared_ptr<CircleNodeOrigin>> origin_table;

  if (_op_table.size() > 0 && _source_table.size() > 0)
  {
    for (auto &kv : _op_table)
    {
      const auto node_id = kv.first;
      const auto &source_ids = kv.second;

      std::vector<std::shared_ptr<CircleNodeOrigin>> origins;
      for (auto source_id : source_ids)
      {
        const auto source_name = _source_table.at(source_id);
        origins.push_back(single_origin(source_id, source_name));
      }

      auto origin = composite_origin(origins);
      if (origin_table.insert({node_id, origin}).second == false)
        throw std::runtime_error("Duplicated origin ID");
    }
  }

  return origin_table;
}

} // namespace luci
