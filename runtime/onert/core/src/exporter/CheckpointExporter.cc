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
  setOptimizerData(train_info, execution);
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
  dst.write(_optimizers.data(), _optimizers.size());
  dst.close();
}

void CheckpointExporter::setReservedData()
{
  // Reserved - 16 bytes
  // - magic number for 2 bytes
  // - schema version for 1 byte
  // - reserved for 1 byte
  // - optimizer data offset for 4 bytes * 2
  // - additional data offset for 4 bytes * 1

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
  ptr += sizeof(SCHEMA_VERSION);

  // Write RESERVED
  const uint8_t RESERVED = 0;
  std::memcpy(ptr, &RESERVED, sizeof(RESERVED));
  ptr += sizeof(RESERVED);

  // Write offsets as default value (zero)
  const uint32_t offset = 0;
  std::memcpy(ptr, &offset, sizeof(offset));
  ptr += sizeof(offset);
  std::memcpy(ptr, &offset, sizeof(offset));
  ptr += sizeof(offset);
  std::memcpy(ptr, &offset, sizeof(offset));
  ptr += sizeof(offset);

  assert(ptr == _reserved.data() + RESERVED_SIZE);
}

void CheckpointExporter::setTensorData(std::unique_ptr<onert::exec::Execution> &exec)
{
  // Tensor Buffers
  // - number of buffers for 4 bytes
  // - 1..N offset for 4 bytes * N
  // - buffers for buf_1 size + buf_2 size + .. buf_N size bytes

  if (_reserved.size() != RESERVED_SIZE)
    throw std::runtime_error{"Invalid reserved buffer"};

  // get Tensor size
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

  const uint32_t count = sizes.size();
  const uint32_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
  const auto buf_size = sizeof(uint32_t) + sizeof(uint32_t) * count + total_size;
  _buffers.resize(buf_size);

  // Pointer to the start address of the buffer
  char *ptr = _buffers.data();

  // Write n_buffers
  std::memcpy(ptr, &count, sizeof(count));
  ptr += sizeof(count);

  // Write offset
  uint32_t buf_offset = _reserved.size() + sizeof(count) + sizeof(uint32_t) * count;
  for (const uint32_t v : sizes)
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

  assert(ptr == _buffers.data() + buf_size);
}

void CheckpointExporter::setOptimizerData(
  std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
  std::unique_ptr<onert::exec::Execution> &exec)
{
  // TODO Support multiple optimizer
  switch (train_info->optimizerInfo().optim_code)
  {
    case onert::ir::train::OptimizerCode::Adam:
      setAdamOptimizerData(exec);
      break;
    default:
      break;
  }
}

void CheckpointExporter::setAdamOptimizerData(std::unique_ptr<onert::exec::Execution> &exec)
{
  // Reserved
  // - optimizer data offset for 4 bytes * 2
  //   - moving average offset 4 bytes
  //   - value offset 4 bytes
  // Optimizer Buffers
  // - 1..M offset for 4 bytes * M
  // - buffers for buf_1 size + buf_2 size + .. buf_M size bytes
  // - 1..V offset for 4 bytes * V
  // - buffers for buf_1 size + buf_2 size + .. buf_V size bytes

  if (_reserved.size() != RESERVED_SIZE)
    throw std::runtime_error{"Invalid reserved buffer"};

  if (_buffers.size() == 0)
    throw std::runtime_error{"Invalid tensor buffer"};

  // get moving average and value size
  std::vector<uint32_t> m_sizes;
  std::vector<uint32_t> v_sizes;
  exec->iterateTrainableTensors(
    [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
      const auto opt_vars = trainable_tensor->optVars();
      assert(opt_vars.size() == 2);
      m_sizes.emplace_back(opt_vars[0]->total_size());
      v_sizes.emplace_back(opt_vars[1]->total_size());
    });

  const auto m_offset_size = sizeof(uint32_t) * m_sizes.size();
  const auto v_offset_size = sizeof(uint32_t) * v_sizes.size();
  const auto m_total_size = std::accumulate(m_sizes.begin(), m_sizes.end(), 0);
  const auto v_total_size = std::accumulate(v_sizes.begin(), v_sizes.end(), 0);
  auto buf_size = m_offset_size + m_total_size + v_offset_size + v_total_size;
  _optimizers.resize(buf_size);

  const uint32_t m_offset = _reserved.size() + _buffers.size();
  uint32_t m_buf_offset = m_offset + m_offset_size;
  // Pointer to the start address of the optimizer buffer
  char *m_ptr = _optimizers.data();

  // Write moving average offset
  for (const uint32_t m : m_sizes)
  {
    std::memcpy(m_ptr, &m_buf_offset, sizeof(m_buf_offset));
    m_ptr += sizeof(m_buf_offset);

    m_buf_offset += m;
  }

  const uint32_t v_offset = m_offset + m_offset_size + m_total_size;
  uint32_t v_buf_offset = v_offset + v_offset_size;
  char *v_ptr = _optimizers.data() + m_offset_size + m_total_size;

  // Write value offset
  for (const uint32_t v : v_sizes)
  {
    std::memcpy(v_ptr, &v_buf_offset, sizeof(v_buf_offset));
    v_ptr += sizeof(v_buf_offset);

    v_buf_offset += v;
  }

  // Write moving average and value buffers
  [[maybe_unused]] auto index = 0;
  exec->iterateTrainableTensors(
    [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
      const auto opt_vars = trainable_tensor->optVars();
      assert(opt_vars.size() == 2);
      assert(m_sizes[index] == opt_vars[0]->total_size());
      assert(v_sizes[index++] == opt_vars[1]->total_size());
      std::memcpy(m_ptr, opt_vars[0]->buffer(), opt_vars[0]->total_size());
      m_ptr += opt_vars[0]->total_size();
      std::memcpy(v_ptr, opt_vars[1]->buffer(), opt_vars[1]->total_size());
      v_ptr += opt_vars[1]->total_size();
    });

  assert(m_ptr == _optimizers.data() + m_offset_size + m_total_size);
  assert(v_ptr == _optimizers.data() + m_offset_size + m_total_size + v_offset_size + v_total_size);

  // Pointer to the start address of the buffer
  char *ptr = _reserved.data();
  ptr += 4; // magic number(2) + schema version(1) + reserved(1)

  // Write Adam M offset
  memcpy(ptr, &m_offset, sizeof(m_offset));
  ptr += sizeof(m_offset);

  // Write Adam V offset
  memcpy(ptr, &v_offset, sizeof(v_offset));
}

} // namespace exporter
} // namespace onert
