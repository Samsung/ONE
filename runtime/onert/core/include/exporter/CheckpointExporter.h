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

#ifndef __ONERT_EXPORTER_CHECKPOINT_EXPORTER_H__
#define __ONERT_EXPORTER_CHECKPOINT_EXPORTER_H__

#include <string>
#include <vector>
#include <memory>

#include "ir/Checkpoint.h"

namespace onert
{
namespace exec
{
class Execution;
} // namespace exec
namespace ir
{
namespace train
{
class TrainingInfo;
} // namespace train
} // namespace ir
} // namespace onert

namespace onert
{
namespace exporter
{

class DataBuffer
{
public:
  /*
   * length: number of tensors
   * total_size: sum of all tensor sizes (in bytes) in the model
   */
  void resize(uint32_t length, uint32_t total_size);

  void setOffset(uint32_t offset);

  void setData(const char *data, uint32_t size);

  void write(std::ofstream &ostream);

private:
  std::vector<uint32_t> _offset;
  std::vector<char> _data;
  uint32_t _cur_offset;

  uint32_t _index;          // current index of data
  char *_data_ptr;          // pointer to the data buffer
  uint32_t _remaining_size; // remaining size of data buffer
};

class CheckpointExporter
{
public:
  CheckpointExporter(const std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
                     const std::unique_ptr<onert::exec::Execution> &exec);

  void save(const std::string &path);

private:
  void setTensorData(const std::unique_ptr<onert::exec::Execution> &exec);

  // void setReservedData();
  // void setTensorData(const std::unique_ptr<onert::exec::Execution> &exec);
  // void setOptimizerData(const std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
  //                       const std::unique_ptr<onert::exec::Execution> &exec);
  // void setAdamOptimizerData(const std::unique_ptr<onert::exec::Execution> &exec);

private:
  checkpoint::Header _header;
  checkpoint::Footer _footer;
  DataBuffer _tensor_data;

  // const uint32_t RESERVED_SIZE = 16;

  // std::vector<char> _reserved;
  // std::vector<char> _buffers;
  // std::vector<char> _optimizers;
};

} // namespace exporter
} // namespace onert

#endif // __ONERT_EXPORTER_CHECKPOINT_EXPORTER_H__
