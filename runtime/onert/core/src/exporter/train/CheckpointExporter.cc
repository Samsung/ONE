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

#include "exporter/train/CheckpointExporter.h"

#include "exec/Execution.h"
#include "ir/train/TrainingInfo.h"
#include "ir/train/Checkpoint.h"

#include <fstream>
#include <iostream>
#include <numeric>

namespace
{

using namespace onert;
using namespace ir;
using namespace train;
using namespace checkpoint;
using namespace exec;

struct DataBuffer
{
  void setSize(const std::vector<uint32_t> &sizes)
  {
    _offset.resize(sizes.size());
    uint32_t total = std::accumulate(sizes.begin(), sizes.end(), 0);
    _data.resize(total);

    _offset_it = _offset.begin();
    _data_ptr = _data.data();
  }

  void setOffset(uint32_t offset) { _start_offset = offset + _offset.size() * sizeof(uint32_t); }

  // This function should be called after executing the setSize() and setOffset() functions.
  void setData(const char *data, uint32_t size)
  {
    assert(_offset_it != _offset.end());
    assert(_data_ptr - _data.data() + size <= _data.size());

    *_offset_it++ = _start_offset;
    if (data && size > 0)
      std::memcpy(_data_ptr, data, size);
    _data_ptr += size;
    _start_offset += size;
  }

  uint32_t size() const { return sizeof(uint32_t) * _offset.size() + _data.size(); }

  void write(std::ofstream &ostream)
  {
    ostream.write(reinterpret_cast<const char *>(&_offset[0]),
                  static_cast<std::streamsize>(sizeof(_offset[0]) * _offset.size()));
    ostream.write(reinterpret_cast<const char *>(&_data[0]),
                  static_cast<std::streamsize>(_data.size()));
  }

private:
  std::vector<uint32_t> _offset;
  std::vector<char> _data;
  uint32_t _start_offset = 0;
  std::vector<uint32_t>::iterator _offset_it;
  char *_data_ptr = nullptr;
};

class CheckpointExporter
{
public:
  CheckpointExporter(const ir::train::TrainingInfo *const train_info,
                     const exec::Execution *const exec)
  {
    std::memset(&_header, 0, sizeof(_header));
    _header.magic = checkpoint::MAGIC_NUMBER;
    _header.schema = checkpoint::SCHEMA_VERSION;

    uint32_t offset = sizeof(_header);

    auto length = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *) { length++; });
    _header.length = length;

    setTensorData(offset, exec);
    offset += _tensor_data.size();

    _header.opt1_offset = offset;
    setOptimizerData(offset, train_info, exec);
    if (_optimizer_data.size() > 2)
      throw std::runtime_error{"Do not support optimizer data more than 2."};
    if (_optimizer_data.size() > 0)
    {
      offset += _optimizer_data[0].size();
      if (_optimizer_data.size() > 1)
      {
        _header.opt2_offset = offset;
        offset += _optimizer_data[1].size();
      }
    }

    _header.other_offset = offset;

    std::memset(&_footer, 0, sizeof(_footer));
    _footer.cur_step = train_info->trainingStep();
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
  void setTensorData(uint32_t start_offset, const exec::Execution *const exec)
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

    _tensor_data.setSize(sizes);
    _tensor_data.setOffset(start_offset);
    [[maybe_unused]] auto vindex = 0;
    exec->iterateTrainableTensors([&](const ir::OperandIndex &,
                                      const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      assert(sizes[vindex++] == tensor->total_size());
      _tensor_data.setData(reinterpret_cast<const char *>(tensor->buffer()), tensor->total_size());
    });
  }

  void setOptimizerData(uint32_t start_offset, const ir::train::TrainingInfo *const train_info,
                        const exec::Execution *const exec)
  {
    // TODO Support multiple optimizer
    switch (train_info->optimizerInfo().optim_code)
    {
      case onert::ir::train::OptimizerCode::Adam:
        setAdamOptimizerData(start_offset, exec);
        break;
      default:
        break;
    }
  }

  void setAdamOptimizerData(uint32_t start_offset, const exec::Execution *const exec)
  {
    // Adam optimizer has two optimizer variables. (mean, variance)
    constexpr auto ADAM_VARIABLE_COUNT = 2;

    std::vector<uint32_t> sizes;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
        const auto &opt_vars = trainable_tensor->optVars();

        // Untrainable tensor should not have any optimizer variables.
        assert(opt_vars.size() == ADAM_VARIABLE_COUNT || opt_vars.size() == 0);

        uint32_t size = 0;
        if (opt_vars.size() == ADAM_VARIABLE_COUNT)
        {
          assert(opt_vars[0]->total_size() == opt_vars[1]->total_size());
          size = opt_vars[0]->total_size();
        }

        sizes.emplace_back(size);
      });

    assert(_header.length == sizes.size());

    _optimizer_data.resize(ADAM_VARIABLE_COUNT);
    for (auto &opt : _optimizer_data)
    {
      opt.setSize(sizes);
      opt.setOffset(start_offset);
      start_offset += opt.size();
    }

    [[maybe_unused]] auto vindex = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
        const auto &opt_vars = trainable_tensor->optVars();

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

} // namespace

namespace onert::exporter::train
{

void exportCheckpoint(const std::string &filename,
                      const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                      const std::unique_ptr<exec::Execution> &exec)
{
  CheckpointExporter exporter(train_info.get(), exec.get());
  exporter.save(filename);
}

} // namespace onert::exporter::train
