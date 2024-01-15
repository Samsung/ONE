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

#include "CircleExportMetadata.h"

#include <luci/UserSettings.h>

namespace
{

void write_u32(std::vector<uint8_t> &to, uint32_t value)
{
  to.emplace_back(0xFF & (value >> 0 * 8));
  to.emplace_back(0xFF & (value >> 1 * 8));
  to.emplace_back(0xFF & (value >> 2 * 8));
  to.emplace_back(0xFF & (value >> 3 * 8));
}

flatbuffers::Offset<circle::Metadata> metadata_offset(flatbuffers::FlatBufferBuilder &builder,
                                                      luci::SerializedModelData &md,
                                                      const std::vector<uint8_t> &data,
                                                      const std::string &metadata_name)
{
  auto buffer_id = static_cast<uint32_t>(md._buffers.size());
  md._buffers.push_back(circle::CreateBufferDirect(builder, &data));
  return circle::CreateMetadataDirect(builder, metadata_name.c_str(), buffer_id);
}

} // namespace

namespace luci
{

// 'execution_plan_table' is encoded to binary format.
const std::vector<uint8_t> CircleExportMetadata::encoded_execution_plan_table()
{
  std::vector<uint8_t> data;

  write_u32(data, _execution_plan_table.size());

  for (auto &kv : _execution_plan_table)
  {
    const auto id = kv.first;
    write_u32(data, id);

    const auto &plan_vector = kv.second;
    const auto size = plan_vector.size();
    write_u32(data, size);

    for (auto elem : plan_vector)
    {
      write_u32(data, elem);
    }
  }

  return data;
}

// 'source_table' is encoded to binary format.
const std::vector<uint8_t> CircleExportMetadata::encoded_source_table(void)
{
  std::vector<uint8_t> data;

  write_u32(data, _source_table.size());

  for (auto &kv : _source_table)
  {
    const auto id = kv.first;
    write_u32(data, id);

    const auto &origin_name = kv.second;
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

    const auto &origins = kv.second;
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

namespace luci
{

std::vector<flatbuffers::Offset<circle::Metadata>>
createCircleMetadataVector(flatbuffers::FlatBufferBuilder &builder, luci::SerializedModelData &md)
{
  std::vector<flatbuffers::Offset<circle::Metadata>> metadata_vec;

  auto settings = luci::UserSettings::settings();
  if (settings->get(luci::UserSettings::Key::ProfilingDataGen))
  {
    metadata_vec.emplace_back(
      metadata_offset(builder, md, md._metadata.encoded_source_table(), "ONE_source_table"));

    metadata_vec.emplace_back(
      metadata_offset(builder, md, md._metadata.encoded_op_table(), "ONE_op_table"));
  }
  if (settings->get(luci::UserSettings::Key::ExecutionPlanGen))
  {
    metadata_vec.emplace_back(metadata_offset(
      builder, md, md._metadata.encoded_execution_plan_table(), "ONE_execution_plan_table"));
  }
  return metadata_vec;
}

} // namespace luci
