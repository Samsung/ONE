/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "exporter/CheckpointExporter.h"

#include "exec/Execution.h"
#include "ir/train/TrainingInfo.h"

#include <fstream>
#include <iostream>

namespace onert
{
namespace exporter
{

CheckpointExporter::CheckpointExporter(std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
                                       std::unique_ptr<onert::exec::Execution> &execution)
{
  uint32_t total_size = getTotalSize();
  _data.resize(total_size);

  // Point to start of the buffer
  char *ptr = _data.data();

  // Write MAGIC NUMBER
  std::memcpy(ptr, &MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
  ptr += sizeof(MAGIC_NUMBER);

  // Write SCHEMA VERSION
  std::memcpy(ptr, &SCHEMA_VERSION, sizeof(SCHEMA_VERSION));
  ptr += sizeof(SCHEMA_VERSION);

  // Reserved
  ptr += sizeof(RESERVED);

  UNUSED_RELEASE(train_info);
  UNUSED_RELEASE(execution);
}

void CheckpointExporter::save(const std::string &path)
{
  std::ofstream dst(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!dst.is_open())
    throw std::runtime_error{"Failed to save checkpoint: " + path};

  dst.write(_data.data(), _data.size());
  dst.close();
}

uint32_t CheckpointExporter::getTotalSize()
{
  uint32_t size = 0;
  size += (sizeof(MAGIC_NUMBER) + sizeof(SCHEMA_VERSION) + sizeof(RESERVED));
  return size;
}

} // namespace exporter
} // namespace onert
