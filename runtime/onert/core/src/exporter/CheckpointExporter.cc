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
#include <numeric>

namespace onert
{
namespace exporter
{

CheckpointExporter::CheckpointExporter(std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
                                       std::unique_ptr<onert::exec::Execution> &execution)
{
  setReservedData();
  setTensorData(execution);

  UNUSED_RELEASE(train_info);
}

void CheckpointExporter::save(const std::string &path)
{
  if (_reserved.size() != RESERVED_SIZE)
    throw std::runtime_error{"Invalid reserved buffer"};

  std::ofstream dst(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!dst.is_open())
    throw std::runtime_error{"Failed to save checkpoint: " + path};

  dst.write(_reserved.data(), _reserved.size());
  dst.write(_buffers.data(), _buffers.size());
  dst.close();
}

void CheckpointExporter::setReservedData()
{
  // Reserved - 16 bytes
  // magic number for 2 bytes
  // schema version for 1 byte
  // reserved for 1 byte
  // offset for 4 * 3 bytes
  // (moving average, value, other parameters offset)

  _reserved.resize(RESERVED_SIZE);

  // Pointer to the start address of the buffer
  char *ptr = _reserved.data();

  // Write MAGIC NUMBER
  const uint16_t MAGIC_NUMBER = 429;
  std::memcpy(ptr, &MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
  ptr += sizeof(MAGIC_NUMBER);

  // Write SCHEMA VERSION
  const uint8_t SCHEMA_VERSION = 1;
  std::memcpy(ptr, &SCHEMA_VERSION, sizeof(SCHEMA_VERSION));
}

void CheckpointExporter::setAdamOffset(uint32_t m_offset, uint32_t v_offset)
{
  if (_reserved.size() != RESERVED_SIZE)
    throw std::runtime_error{"Invalid reserved buffer"};

  // Pointer to the start address of the buffer
  char *ptr = _reserved.data();
  ptr += 4; // magic number(2) + schema version(1) + reserved(1)

  // Write Adam M offset
  memcpy(ptr, &m_offset, sizeof(m_offset));
  ptr += sizeof(m_offset);

  // Write Adam V offset
  memcpy(ptr, &v_offset, sizeof(v_offset));
}

void CheckpointExporter::setTensorData(std::unique_ptr<onert::exec::Execution> &exec)
{
  // Tensor Buffers
  // number of buffers for 4 bytes
  // 1..N offset for 4 * N bytes
  // buffers for buf_1 size + buf_2 size + .. buf_N size bytes

  // get Tensor count
  std::vector<uint32_t> sizes;
  exec->iterateTrainableTensors(
    [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      if (tensor->total_size() >= std::numeric_limits<uint32_t>::max())
      {
        throw std::runtime_error{"Tensor size exceeds the uint32_t max value. This mode does not "
                                 "support saving as a checkpoint file."};
      }

      sizes.emplace_back(tensor->total_size());
    });

  uint32_t count = sizes.size();
  uint32_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
  auto buf_size = sizeof(uint32_t) + sizeof(uint32_t) * count + total_size;
  _buffers.resize(buf_size);

  // Pointer to the start address of the buffer
  char *ptr = _buffers.data();

  // Write n_buffers
  std::memcpy(ptr, &count, sizeof(count));
  ptr += sizeof(count);

  // Write offset
  uint32_t buf_offset = RESERVED_SIZE + sizeof(count) + sizeof(uint32_t) * count;
  for (uint32_t v : sizes)
  {
    std::memcpy(ptr, &buf_offset, sizeof(buf_offset));
    ptr += sizeof(buf_offset);

    buf_offset += v;
  }

  // Write tensor buffers
  [[maybe_unused]] auto vindex = 0;
  exec->iterateTrainableTensors(
    [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      assert(sizes[vindex++] == tensor->total_size());
      std::memcpy(ptr, tensor->buffer(), tensor->total_size());
      ptr += tensor->total_size();
    });
}

} // namespace exporter
} // namespace onert
