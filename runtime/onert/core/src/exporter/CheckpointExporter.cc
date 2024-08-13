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

class DataBuffer
{
public:
  /*
   * length: number of tensors
   * total_size: sum of all tensor sizes (in bytes) in the model
   */
  void resize(uint32_t length, uint32_t total_size)
  {
    _offset.resize(length);
    _data.resize(total_size);
    _index = 0;
    _data_ptr = _data.data();
    _remaining_size = _data.size();
  }

  void setOffset(uint32_t offset) { _cur_offset = offset + sizeof(uint32_t) * _offset.size(); }

  void setData(const char *data, uint32_t size)
  {
    assert(_index < _offset.size());
    assert(size <= _remaining_size);
    _offset[_index++] = _cur_offset;
    if (size > 0 && data)
      std::memcpy(_data_ptr, data, size);
    _cur_offset += size;
    _data_ptr += size;
    _remaining_size -= size;
  }

  void write(std::ofstream &ostream)
  {
    ostream.write(reinterpret_cast<const char *>(&_offset[0]), sizeof(uint32_t) * _offset.size());
    ostream.write(reinterpret_cast<const char *>(&_data[0]), _data.size());
  }

  uint32_t size() const { return sizeof(uint32_t) * _offset.size() + _data.size(); }

private:
  std::vector<uint32_t> _offset;
  std::vector<char> _data;

  uint32_t _cur_offset;     // current offset from checkpoint buffer
  uint32_t _index;          // current index of data
  char *_data_ptr;          // pointer to the data buffer
  uint32_t _remaining_size; // remaining size of data buffer
};

class CheckpointExporter
{
public:
  CheckpointExporter(const std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
                     const std::unique_ptr<onert::exec::Execution> &exec)
  {
    uint32_t offset = 0;
    std::memset(&_header, 0, sizeof(_header));
    _header.magic = checkpoint::MAGIC_NUMBER;
    _header.schema = checkpoint::SCHEMA_VERSION;
    offset += sizeof(_header);

    auto length = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *) { length++; });
    _header.length = length;

    setTensorData(offset, exec);
    offset += _tensor_data.size();

    _header.opt1_offset = offset;
    // opt2_offset of _header will be stored in setOptimzierData function.
    setOptimizerData(offset, train_info, exec);
    for (const auto &opt : _optimizer_data)
      offset += opt.size();

    _header.other_offset = offset;
    _footer.cur_step = train_info->trainingStep();
    // TODO Set current epoch
    _footer.cur_epoch = 0;
  }

  void save(const std::string &path)
  {
    std::ofstream dst(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!dst.is_open())
      throw std::runtime_error{"Failed to save checkpoint: " + path};

    dst.write(reinterpret_cast<const char *>(&_header), sizeof(_header));
    _tensor_data.write(dst);
    for (auto &opt : _optimizer_data)
      opt.write(dst);
    dst.write(reinterpret_cast<const char *>(&_footer), sizeof(_footer));
    dst.close();
  }

private:
  void setTensorData(uint32_t offset, const std::unique_ptr<onert::exec::Execution> &exec)
  {
    std::vector<uint32_t> sizes;
    exec->iterateTrainableTensors([&](const ir::OperandIndex &,
                                      const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      if (tensor->total_size() >= std::numeric_limits<uint32_t>::max())
      {
        throw std::runtime_error{"Tensor size exceeds the uint32_t max value. This model does not "
                                 "support saving as a checkpoint file."};
      }

      sizes.emplace_back(tensor->total_size());
    });

    assert(_header.length == sizes.size());

    const uint32_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
    _tensor_data.resize(sizes.size(), total_size);

    _tensor_data.setOffset(offset);

    [[maybe_unused]] auto vindex = 0;
    exec->iterateTrainableTensors([&](const ir::OperandIndex &,
                                      const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      assert(sizes[vindex++] == tensor->total_size());
      _tensor_data.setData(reinterpret_cast<const char *>(tensor->buffer()), tensor->total_size());
    });
  }

  void setOptimizerData(uint32_t offset,
                        const std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
                        const std::unique_ptr<onert::exec::Execution> &exec)
  {
    // TODO Support multiple optimizer
    switch (train_info->optimizerInfo().optim_code)
    {
      case onert::ir::train::OptimizerCode::Adam:
        setAdamOptimizerData(offset, exec);
        break;
      default:
        break;
    }
  }

  void setAdamOptimizerData(uint32_t offset, const std::unique_ptr<onert::exec::Execution> &exec)
  {
    // Adam optimizer has two optimizer variables. (mean, variance)
    constexpr auto ADAM_VARIABLE_COUNT = 2;

    std::vector<uint32_t> sizes;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
        const auto opt_vars = trainable_tensor->optVars();

        // Untrainable tensor should not have any optimizer variables.
        assert(opt_vars.size() == ADAM_VARIABLE_COUNT || opt_vars.size() == 0);

        uint32_t size = 0;
        if (opt_vars.size() == ADAM_VARIABLE_COUNT)
        {
          // The sizes of mean and variance are the same.
          assert(opt_vars[0]->total_size() == opt_vars[1]->total_size());
          size = opt_vars[0]->total_size();
        }

        sizes.emplace_back(size);
      });

    assert(_header.length == sizes.size());

    _optimizer_data.resize(ADAM_VARIABLE_COUNT);

    const uint32_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
    for (auto &opt_data : _optimizer_data)
    {
      opt_data.resize(sizes.size(), total_size);
    }

    _header.opt2_offset = offset + _optimizer_data[0].size();

    _optimizer_data[0].setOffset(_header.opt1_offset);
    _optimizer_data[1].setOffset(_header.opt2_offset);

    [[maybe_unused]] auto vindex = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
        const auto opt_vars = trainable_tensor->optVars();

        // Untrainable tensor should not have any optimizer variables.
        assert(opt_vars.size() == ADAM_VARIABLE_COUNT || opt_vars.size() == 0);

        for (auto i = 0; i < ADAM_VARIABLE_COUNT; ++i)
        {
          if (opt_vars.size() == ADAM_VARIABLE_COUNT)
          {
            assert(opt_vars[i]->total_size() == sizes[vindex]);
            _optimizer_data[i].setData(reinterpret_cast<const char *>(opt_vars[i]->buffer()),
                                       opt_vars[i]->total_size());
          }
          else
            _optimizer_data[i].setData(nullptr, 0);
        }
        vindex++;
      });
  }

private:
  checkpoint::Header _header;
  checkpoint::Footer _footer;
  DataBuffer _tensor_data;
  std::vector<DataBuffer> _optimizer_data;
};

void exportCheckpoint(const std::string &filename,
                      const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                      const std::unique_ptr<onert::exec::Execution> &exec)
{
  CheckpointExporter exporter(train_info, exec);
  exporter.save(filename);
}

} // namespace exporter
} // namespace onert
