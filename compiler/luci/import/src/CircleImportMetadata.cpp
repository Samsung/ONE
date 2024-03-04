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

#include "CircleImportMetadata.h"

#include <vector>

namespace
{

template <typename VECTORTYPE> uint32_t read_u32(const VECTORTYPE &buffer, uint32_t idx)
{
  static_assert(std::is_same<typename VECTORTYPE::value_type, uint8_t>::value, "Types mismatch!");

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

// 'source_table' is decoded to std::map<uint32_t, std::string> format.
template <typename VECTORTYPE>
const std::map<uint32_t, std::string> decoded_source_table(const VECTORTYPE &source_table_data)
{
  static_assert(std::is_same<typename VECTORTYPE::value_type, uint8_t>::value, "Types mismatch!");

  std::map<uint32_t, std::string> source_id_name_map;
  uint32_t idx = 0;

  if (source_table_data.size() < 4)
    throw std::runtime_error("Source table decode error : invalid entry number");

  uint32_t entry_number = read_u32(source_table_data, idx);
  idx += sizeof(uint32_t);

  while (idx < source_table_data.size())
  {
    if (idx + 2 * sizeof(uint32_t) > source_table_data.size())
      throw std::runtime_error("Source table decode error : invalid entry item");

    uint32_t id = read_u32(source_table_data, idx);
    idx += sizeof(uint32_t);

    uint32_t length = read_u32(source_table_data, idx);
    idx += sizeof(uint32_t);

    if (idx + sizeof(char) * length > source_table_data.size())
      throw std::runtime_error("Source table decode error : invalid entry data");

    // The last character of name is '\0'.
    // However, as std::string do not use '\0' for finding the end of string,
    // we ignore the character and do not include it in the string.
    std::string origin_name;
    for (uint32_t j = 0; j < length - 1; ++j)
      origin_name += source_table_data.at(idx + j);
    assert(source_table_data.at(idx + length - 1) == '\0');
    idx += sizeof(char) * length;

    if (source_id_name_map.insert({id, origin_name}).second == false)
      throw std::runtime_error("Source table decode error : duplicated origin ID");
  }

  if (idx != source_table_data.size())
    throw std::runtime_error("Source table decode error : data size invalid");

  if (source_id_name_map.size() != entry_number)
    throw std::runtime_error("Source table decode error : result size mismatch");

  return source_id_name_map;
}

// 'op_table' is decoded to std::map<uint32_t, std::set<uint32_t>> format.
template <typename VECTORTYPE>
const std::map<uint32_t, std::set<uint32_t>> decoded_op_table(const VECTORTYPE &op_table_data)
{
  static_assert(std::is_same<typename VECTORTYPE::value_type, uint8_t>::value, "Types mismatch!");

  std::map<uint32_t, std::set<uint32_t>> node_source_ids_map;
  uint32_t idx = 0;

  if (op_table_data.size() < 4)
    throw std::runtime_error("Op table decode error : invalid entry number");

  uint32_t entry_number = read_u32(op_table_data, idx);
  idx += sizeof(uint32_t);

  while (idx < op_table_data.size())
  {
    if (idx + 2 * sizeof(uint32_t) > op_table_data.size())
      throw std::runtime_error("Op table decode error : invalid entry item");

    uint32_t id = read_u32(op_table_data, idx);
    idx += sizeof(uint32_t);

    uint32_t node_num = read_u32(op_table_data, idx);
    idx += sizeof(uint32_t);

    if (idx + sizeof(uint32_t) * node_num > op_table_data.size())
      throw std::runtime_error("Source table decode error : invalid entry data");

    std::set<uint32_t> source_ids;
    for (uint32_t j = 0; j < node_num; ++j)
    {
      uint32_t origin = read_u32(op_table_data, idx);
      idx += sizeof(uint32_t);

      source_ids.insert(origin);
    }

    if (node_source_ids_map.insert({id, source_ids}).second == false)
      throw std::runtime_error("Op table decode error : duplicated origin ID");
  }

  if (idx != op_table_data.size())
    throw std::runtime_error("Op table decode error : data size invalid");

  if (node_source_ids_map.size() != entry_number)
    throw std::runtime_error("Op table decode error : entry number invalid");

  return node_source_ids_map;
}

// 'map_tensors_indexes' is decoded to std::map<uint32_t, uint32_t> format.
template <typename VECTORTYPE>
const std::map<uint32_t, uint32_t> decoded_map_tensors_indexes(const VECTORTYPE &map_tensors_indexes_data)
{
  static_assert(std::is_same<typename VECTORTYPE::value_type, uint8_t>::value, "Types mismatch!");

  std::map<uint32_t, uint32_t> map_tensors_indexes;
  uint32_t idx = 0;

  if (map_tensors_indexes_data.size() < 4)
    throw std::runtime_error("Op table decode error : invalid entry number");

  uint32_t entry_number = read_u32(map_tensors_indexes_data, idx);
  idx += sizeof(uint32_t);

  while (idx < map_tensors_indexes_data.size())
  {
    if (idx + 2 * sizeof(uint32_t) > map_tensors_indexes_data.size())
      throw std::runtime_error("Op table decode error : invalid entry item");

    uint32_t id = read_u32(map_tensors_indexes_data, idx);
    idx += sizeof(uint32_t);

    uint32_t origin_id = read_u32(map_tensors_indexes_data, idx);
    idx += sizeof(uint32_t);

    if (map_tensors_indexes.insert({id, origin_id}).second == false)
      throw std::runtime_error("Op table decode error : duplicated origin ID");

//    uint32_t size = read_u32(execution_plan_data, idx);
//
//    if (size == 0)
//      throw std::runtime_error("Op table decode error : empty execution plan entry");
//
//    idx += sizeof(uint32_t);
//
//    if (idx + sizeof(uint32_t) * size > execution_plan_data.size())
//      throw std::runtime_error("Source table decode error : invalid entry data");
//
//    std::vector<uint32_t> execution_plan_vector;
//    for (uint32_t j = 0; j < size; ++j)
//    {
//      uint32_t execution_plan_inform = read_u32(execution_plan_data, idx);
//      idx += sizeof(uint32_t);
//
//      execution_plan_vector.push_back(execution_plan_inform);
//    }
//
//    if (execution_plan_table.insert({id, execution_plan_vector}).second == false)
//      throw std::runtime_error("Op table decode error : duplicated origin ID");
  }

  if (idx != map_tensors_indexes_data.size())
    throw std::runtime_error("Op table decode error : data size invalid");

  if (map_tensors_indexes.size() != entry_number)
    throw std::runtime_error("Op table decode error : entry number invalid");

  return map_tensors_indexes;
}

// 'execution_plan_table' is decoded to std::map<uint32_t, std::vector<uint32_t>> format.
template <typename VECTORTYPE>
const luci::ExecutionPlanTable decoded_execution_plan(const VECTORTYPE &execution_plan_data)
{
  static_assert(std::is_same<typename VECTORTYPE::value_type, uint8_t>::value, "Types mismatch!");

  luci::ExecutionPlanTable execution_plan_table;
  uint32_t idx = 0;

  if (execution_plan_data.size() < 4)
    throw std::runtime_error("Op table decode error : invalid entry number");

  uint32_t entry_number = read_u32(execution_plan_data, idx);
  idx += sizeof(uint32_t);

  while (idx < execution_plan_data.size())
  {
    if (idx + 2 * sizeof(uint32_t) > execution_plan_data.size())
      throw std::runtime_error("Op table decode error : invalid entry item");

    uint32_t id = read_u32(execution_plan_data, idx);
    idx += sizeof(uint32_t);

    uint32_t size = read_u32(execution_plan_data, idx);

    if (size == 0)
      throw std::runtime_error("Op table decode error : empty execution plan entry");

    idx += sizeof(uint32_t);

    if (idx + sizeof(uint32_t) * size > execution_plan_data.size())
      throw std::runtime_error("Source table decode error : invalid entry data");

    std::vector<uint32_t> execution_plan_vector;
    for (uint32_t j = 0; j < size; ++j)
    {
      uint32_t execution_plan_inform = read_u32(execution_plan_data, idx);
      idx += sizeof(uint32_t);

      execution_plan_vector.push_back(execution_plan_inform);
    }

    if (execution_plan_table.insert({id, execution_plan_vector}).second == false)
      throw std::runtime_error("Op table decode error : duplicated origin ID");
  }

  if (idx != execution_plan_data.size())
    throw std::runtime_error("Op table decode error : data size invalid");

  if (execution_plan_table.size() != entry_number)
    throw std::runtime_error("Op table decode error : entry number invalid");

  return execution_plan_table;
}

} // namespace

namespace luci
{

CircleImportMetadata::CircleImportMetadata(const luci::CircleReader &reader)
{
  const auto metadata = reader.metadata();
  for (uint32_t i = 0; i < metadata.size(); ++i)
  {
    const auto *meta = metadata[i];
    assert(meta != nullptr);

    assert(meta->buffer() < reader.buffers().size());
    assert(reader.buffers()[meta->buffer()] != nullptr);
    const auto buffer = luci::wrap(reader.buffers()[meta->buffer()]->data());

    assert(meta->name() != nullptr);
    if (meta->name()->str().compare("ONE_op_table") == 0)
      _op_table = decoded_op_table(buffer);
    else if (meta->name()->str().compare("ONE_source_table") == 0)
      _source_table = decoded_source_table(buffer);
    else if (meta->name()->str().compare("ONE_execution_plan_table") == 0)
      _execution_plan_table = decoded_execution_plan(buffer);
    else if (meta->name()->str().compare("ONE_train_inform") == 0)
      _map_tensors_indexes = decoded_map_tensors_indexes(buffer);
  }
}

const OriginTable CircleImportMetadata::origin_table(void)
{
  OriginTable origin_table;

  if (_op_table.size() > 0 && _source_table.size() > 0)
  {
    for (auto &kv : _op_table)
    {
      const auto node_id = kv.first;
      const auto &source_ids = kv.second;

      std::vector<std::shared_ptr<CircleNodeOrigin>> origins;
      for (auto source_id : source_ids)
      {
        const auto &source_name = _source_table.at(source_id);
        origins.push_back(single_origin(source_id, source_name));
      }

      auto origin = composite_origin(origins);
      origin_table.emplace(node_id, origin);
    }
  }

  return origin_table;
}

} // namespace luci
