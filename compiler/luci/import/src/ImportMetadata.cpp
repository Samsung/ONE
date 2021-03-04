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

#include "ImportMetadata.h"

#include <vector>

namespace
{

/**
 * 'op_table'
 * - [ id : uint32_t ][ node_num : uint32_t ][ node_ids : node_num * uint32_t ]
 * - 'node_num' should be larger than zero. (TODO Check it!)
 */
std::map<uint32_t, std::set<uint32_t>> metadata_with_op_table(const std::vector<uint8_t> &buffer)
{
  std::map<uint32_t, std::set<uint32_t>> node_source_ids_map;

  uint32_t idx = 0;
  while (idx < buffer.size())
  {
    uint32_t id = 0;
    id += (buffer.at(idx + 0) << 0 * 8);
    id += (buffer.at(idx + 1) << 1 * 8);
    id += (buffer.at(idx + 2) << 2 * 8);
    id += (buffer.at(idx + 3) << 3 * 8);
    idx += 4;

    uint32_t node_num = 0;
    node_num += (buffer.at(idx + 0) << 0 * 8);
    node_num += (buffer.at(idx + 1) << 1 * 8);
    node_num += (buffer.at(idx + 2) << 2 * 8);
    node_num += (buffer.at(idx + 3) << 3 * 8);
    idx += 4;

    std::set<uint32_t> source_ids;
    for (uint32_t j = 0; j < node_num; ++j)
    {
      uint32_t origin = 0;
      origin += (buffer.at(idx + 4 * j + 0) << 0 * 8);
      origin += (buffer.at(idx + 4 * j + 1) << 1 * 8);
      origin += (buffer.at(idx + 4 * j + 2) << 2 * 8);
      origin += (buffer.at(idx + 4 * j + 3) << 3 * 8);

      source_ids.insert(origin);
    }
    idx += (node_num * 4);

    if (node_source_ids_map.insert({id, source_ids}).second == false)
      throw std::runtime_error("Duplicated origin ID");
  }
  assert(idx == buffer.size());

  return node_source_ids_map;
}

/**
 * 'source_table'
 * - [ id : uint32_t ][ length : uint32_t ][ data : 'length' Bytes ]
 */
std::map<uint32_t, std::string> metadata_with_source_table(const std::vector<uint8_t> &buffer)
{
  std::map<uint32_t, std::string> source_id_name_map;

  uint32_t idx = 0;
  while (idx < buffer.size())
  {
    uint32_t id = 0;
    id += (buffer.at(idx + 0) << 0 * 8);
    id += (buffer.at(idx + 1) << 1 * 8);
    id += (buffer.at(idx + 2) << 2 * 8);
    id += (buffer.at(idx + 3) << 3 * 8);
    idx += 4;

    uint32_t length = 0;
    length += (buffer.at(idx + 0) << 0 * 8);
    length += (buffer.at(idx + 1) << 1 * 8);
    length += (buffer.at(idx + 2) << 2 * 8);
    length += (buffer.at(idx + 3) << 3 * 8);
    idx += 4;

    std::string origin_name;
    for (uint32_t j = 0; j < length; ++j)
      origin_name += (char)buffer.at(idx + j);
    idx += length;

    if (source_id_name_map.insert({id, origin_name}).second == false)
      throw std::runtime_error("Duplicated origin ID");
  }
  assert(idx == buffer.size());

  return source_id_name_map;
}

} // namespace

namespace luci
{

ImportMetadata::ImportMetadata(const luci::CircleReader &reader)
{
  std::map<uint32_t, std::set<uint32_t>> node_source_ids_map;
  std::map<uint32_t, std::string> source_id_name_map;

  const auto &metadata = reader.metadata();

  for (uint32_t i = 0; i < metadata.size(); ++i)
  {
    const circle::MetadataT &meta = *metadata[i];

    assert(meta.buffer < reader.buffers().size());
    const std::vector<uint8_t> &buffer = reader.buffers()[meta.buffer]->data;

    // TODO Handle this by enum
    if (meta.name.compare("op_table") == 0)
    {
      node_source_ids_map = metadata_with_op_table(buffer);
    }
    else if (meta.name.compare("source_table") == 0)
    {
      source_id_name_map = metadata_with_source_table(buffer);
    }
  }

  // Create Origin Metadata
  if (node_source_ids_map.size() > 0 && source_id_name_map.size() > 0)
  {
    for (auto &kv : node_source_ids_map)
    {
      const auto node_id = kv.first;
      const auto &source_ids = kv.second;

      std::vector<std::shared_ptr<CircleNodeOrigin>> origins;
      for (auto source_id : source_ids)
      {
        const auto source_name = source_id_name_map.at(source_id);
        origins.push_back(single_origin(source_id, source_name));
      }

      auto origin = composite_origin(origins);
      if (_origin_table.insert({node_id, origin}).second == false)
        throw std::runtime_error("Duplicated origin ID");
    }
  }
}

} // namespace luci
