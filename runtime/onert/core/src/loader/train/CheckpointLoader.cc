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

#include "loader/train/CheckpointLoader.h"

#include "exec/Execution.h"
#include "ir/train/Checkpoint.h"
#include "ir/train/TrainingInfo.h"

#include <filesystem>
#include <fstream>

namespace
{

using namespace onert;
using namespace ir;
using namespace train;
using namespace checkpoint;
using namespace exec;

struct DataBufferPair
{
  uint32_t offset;
  uint32_t size;
};

class DataBuffer
{
public:
  DataBuffer() = default;
  DataBuffer(uint32_t size) { setSize(size); }

  void setSize(uint32_t size)
  {
    _offset.resize(size);
    _size.resize(size);
  }

  char *getOffsetBuf() { return reinterpret_cast<char *>(_offset.data()); }

  // This function should be called after loading the _offset buffer.
  void calculateSize(uint32_t next_start_offset)
  {
    assert(_offset.size() == _size.size());
    for (size_t i = 0; i < _offset.size() - 1; ++i)
      _size[i] = _offset[i + 1] - _offset[i];
    _size.back() = next_start_offset - _offset.back();
  }

  // offset, size
  DataBufferPair operator[](uint32_t i) const
  {
    assert(_offset.size() == _size.size());
    assert(i < _offset.size());
    return DataBufferPair{_offset[i], _size[i]};
  }

private:
  std::vector<uint32_t> _offset;
  std::vector<uint32_t> _size;
};

class CheckpointLoader final
{
public:
  CheckpointLoader(const std::string &filename)
  {
    if (filename.empty() || !std::filesystem::exists(filename))
      throw std::runtime_error{"Invalid checkpoint file"};

    _file.open(filename.c_str(), std::ios::binary | std::ios::in);
    if (!_file.good())
      throw std::runtime_error{"Failed to open checkpoint file"};

    _file.seekg(0, std::ios::end);
    const unsigned long filesize = _file.tellg();
    _file.seekg(0, std::ios::beg);

    if (filesize < sizeof(_header))
      throw std::runtime_error{"Invalid checkpoint file data"};

    memset(reinterpret_cast<char *>(&_header), 0, sizeof(_header));
    _file.read(reinterpret_cast<char *>(&_header), sizeof(_header));
    if (_file.fail())
      throw std::runtime_error{"Failed to load header data"};

    if (_header.magic != checkpoint::MAGIC_NUMBER)
      throw std::runtime_error{"Invalid MAGIC NUMBER"};

    if (_header.schema != checkpoint::SCHEMA_VERSION)
      throw std::runtime_error{"Invalid SCHEMA VERSION"};

    _tensor_data.setSize(_header.length);
    _file.read(_tensor_data.getOffsetBuf(),
               static_cast<std::streamsize>(_header.length * sizeof(uint32_t)));
    if (_file.fail())
      throw std::runtime_error{"Failed to load tensor data"};
    _tensor_data.calculateSize(_header.opt1_offset);

    if (_header.opt1_offset)
    {
      DataBuffer opt1_data(_header.length);
      _file.seekg(static_cast<std::streamoff>(_header.opt1_offset), std::ios::beg);
      _file.read(opt1_data.getOffsetBuf(),
                 static_cast<std::streamsize>(_header.length * sizeof(uint32_t)));
      opt1_data.calculateSize(_header.opt2_offset);
      _optimizer_data.emplace_back(std::move(opt1_data));
    }

    if (_header.opt2_offset)
    {
      DataBuffer opt2_data(_header.length);
      _file.seekg(static_cast<std::streamoff>(_header.opt2_offset), std::ios::beg);
      _file.read(opt2_data.getOffsetBuf(),
                 static_cast<std::streamsize>(_header.length * sizeof(uint32_t)));
      opt2_data.calculateSize(_header.other_offset);
      _optimizer_data.emplace_back(std::move(opt2_data));
    }

    if ((filesize - _header.other_offset) != sizeof(_footer))
      throw std::runtime_error{"Invalid checkpoint file footer data"};

    memset(reinterpret_cast<char *>(&_footer), 0, sizeof(_footer));
    _file.seekg(static_cast<std::streamoff>(_header.other_offset), std::ios::beg);
    _file.read(reinterpret_cast<char *>(&_footer), sizeof(_footer));
  }

  ~CheckpointLoader()
  {
    if (_file.is_open())
      _file.close();
  }

  void updateTensor(const std::unique_ptr<exec::Execution> &exec)
  {
    uint32_t vindex = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *) { vindex++; });

    if (_header.length != vindex)
      throw std::runtime_error{
        "Invalid number of tensors between TrainingInfo and checkpoint file"};

    // Reset EOF bit
    _file.clear();

    vindex = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        assert(_tensor_data[vindex].size == tensor->total_size());
        _file.seekg(static_cast<std::streamoff>(_tensor_data[vindex].offset), std::ios::beg);
        _file.read(reinterpret_cast<char *>(tensor->buffer()),
                   static_cast<std::streamsize>(tensor->total_size()));
        vindex++;
      });
  }

  void updateOptimizer(const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                       const std::unique_ptr<onert::exec::Execution> &exec)
  {
    ir::train::OptimizerCode ckpt_opt_code = ir::train::OptimizerCode::SGD;
    // TODO Support other optimizers
    if (_optimizer_data.size() == 2)
      ckpt_opt_code = ir::train::OptimizerCode::Adam;

    if (ckpt_opt_code != train_info->optimizerInfo().optim_code)
      throw std::runtime_error{
        "Not compatible optimizer type between TrainingInfo and checkpoint file"};

    switch (train_info->optimizerInfo().optim_code)
    {
      case ir::train::OptimizerCode::Adam:
        updateAdamOptimizer(exec);
        break;
      default:
        break;
    }
  }

  void updateTrainingInfo(const std::unique_ptr<ir::train::TrainingInfo> &train_info)
  {
    // TODO Verify cur_step value
    train_info->trainingStep() = _footer.cur_step;
  }

private:
  void updateAdamOptimizer(const std::unique_ptr<onert::exec::Execution> &exec)
  {
    // Adam optimizer has two optimizer variables. (mean, variance)
    [[maybe_unused]] const std::size_t ADAM_VARIABLE_COUNT = 2;

    // Reset EOF bit
    _file.clear();

    uint32_t vindex = 0;
    exec->iterateTrainableTensors([&](const ir::OperandIndex &,
                                      const backend::train::ITrainableTensor *tensor) {
      assert(tensor);
      auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
      const auto opt_vars = trainable_tensor->optVars();

      // Untrainable tensor should not have any optimizer variables.
      assert(opt_vars.size() == ADAM_VARIABLE_COUNT || opt_vars.size() == 0);

      for (size_t i = 0; i < opt_vars.size(); ++i)
      {
        assert(opt_vars[i]->total_size() == _optimizer_data[i][vindex].size);
        _file.seekg(static_cast<std::streamoff>(_optimizer_data[i][vindex].offset), std::ios::beg);
        _file.read(reinterpret_cast<char *>(opt_vars[i]->buffer()),
                   static_cast<std::streamsize>(opt_vars[i]->total_size()));
      }
      vindex++;
    });
  }

private:
  std::ifstream _file;
  checkpoint::Header _header;
  checkpoint::Footer _footer;
  DataBuffer _tensor_data;
  std::vector<DataBuffer> _optimizer_data;
};

} // namespace

namespace onert::loader::train
{

void loadCheckpoint(const std::string &filename,
                    const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                    const std::unique_ptr<onert::exec::Execution> &exec)
{
  CheckpointLoader loader(filename);
  loader.updateTensor(exec);
  loader.updateOptimizer(train_info, exec);
  loader.updateTrainingInfo(train_info);
}

} // namespace onert::loader::train
