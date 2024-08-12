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

#include "loader/CheckpointLoader.h"

#include "ir/Model.h"
#include "ir/Checkpoint.h"
#include "ir/train/TrainingInfo.h"
#include "exec/Execution.h"
#include "util/Utils.h"

#include <fstream>
#include <filesystem>

namespace onert
{
namespace loader
{
struct DataBufferPair
{
  DataBufferPair(uint32_t _offset, uint32_t _size) : offset{_offset}, size{_size}
  {
    // DO NOTHING
  }

  uint32_t offset;
  uint32_t size;
};

class DataBuffer
{
public:
  void resize(uint32_t length)
  {
    _offset.resize(length);
    _size.resize(length);
  }

  size_t size() const
  {
    assert(_offset.size() == _size.size());
    return _offset.size();
  }

  char *getOffsetBuf() { return reinterpret_cast<char *>(_offset.data()); }

  void calculateSize(uint32_t next_beg_offset)
  {
    assert(_offset.size() == _size.size());
    uint32_t cur = _offset[0];
    for (size_t i = 1; i < _offset.size(); ++i)
    {
      _size[i - 1] = _offset[i] - cur;
      cur = _offset[i];
    }
    _size.back() = next_beg_offset - _offset.back();
  }

  // offset, size
  DataBufferPair operator[](uint32_t i) const
  {
    assert(_offset.size() == _size.size());
    assert(i <= _offset.size());
    return DataBufferPair{_offset[i], _size[i]};
  }

private:
  std::vector<uint32_t> _offset;
  std::vector<uint32_t> _size;
};

class CheckpointLoader
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
    const auto filesize = _file.tellg();
    _file.seekg(0, std::ios::beg);

    if (filesize < static_cast<long int>(sizeof(_header)))
      throw std::runtime_error{"Invalid checkpoint file data"};

    memset(reinterpret_cast<char *>(&_header), 0, sizeof(_header));
    _file.read(reinterpret_cast<char *>(&_header), sizeof(_header));
    if (_file.fail())
      throw std::runtime_error{"Failed to load header data"};

    if (_header.magic != checkpoint::MAGIC_NUMBER)
      throw std::runtime_error{"Invalid MAGIC NUMBER"};

    if (_header.schema != checkpoint::SCHEMA_VERSION)
      throw std::runtime_error{"Invalid SCHEMA VERSION"};

    _tensor_data.resize(_header.length);
    _file.read(_tensor_data.getOffsetBuf(), _header.length * sizeof(uint32_t));
    _tensor_data.calculateSize(_header.opt1_offset);

    if (_header.opt1_offset)
    {
      _opt1_data.resize(_header.length);
      _file.seekg(_header.opt1_offset, std::ios::beg);
      _file.read(_opt1_data.getOffsetBuf(), _header.length * sizeof(uint32_t));
      _opt1_data.calculateSize(_header.opt2_offset);
    }

    if (_header.opt2_offset)
    {
      _opt2_data.resize(_header.length);
      _file.seekg(_header.opt2_offset, std::ios::beg);
      _file.read(_opt2_data.getOffsetBuf(), _header.length * sizeof(uint32_t));
      _opt2_data.calculateSize(_header.other_offset);
    }

    // if (filesize - static_cast<long int>(_header.other_offset) != sizeof(_footer))
    //   throw std::runtime_error{"Invalid checkpoint file footer data"};

    // memset(reinterpret_cast<char *>(&_footer), 0, sizeof(_footer));
    // _file.seekg(_header.other_offset, std::ios::beg);
    // _file.read(reinterpret_cast<char *>(&_footer), sizeof(_footer));
  }

  ~CheckpointLoader()
  {
    if (_file.is_open())
      _file.close();
  }

  void updateTensor(const std::unique_ptr<onert::exec::Execution> &exec)
  {
    auto vindex = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *) { vindex++; });

    if (_header.length != vindex)
      throw std::runtime_error{
        "Invalid number of tensors between TrainingInfo and checkpoint file"};

    vindex = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        assert(tensor->total_size() == _tensor_data[vindex].size);
        _file.seekg(_tensor_data[vindex].offset, std::ios::beg);
        _file.read(reinterpret_cast<char *>(tensor->buffer()), tensor->total_size());
        vindex++;
      });
  }

  void updateOptimizer(const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                       const std::unique_ptr<onert::exec::Execution> &exec)
  {
    ir::train::OptimizerCode ckpt_opt_code = ir::train::OptimizerCode::SGD;
    if (_opt1_data.size() > 0 && _opt2_data.size() > 0)
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
    auto vindex = 0;
    exec->iterateTrainableTensors(
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        auto trainable_tensor = const_cast<backend::train::ITrainableTensor *>(tensor);
        const auto opt_vars = trainable_tensor->optVars();
        assert(opt_vars.size() == 2);

        // moving average
        assert(opt_vars[0]->total_size() == _opt1_data[vindex].size);
        _file.seekg(_opt1_data[vindex].offset, std::ios::beg);
        _file.read(reinterpret_cast<char *>(opt_vars[0]->buffer()), opt_vars[0]->total_size());

        // value
        assert(opt_vars[1]->total_size() == _opt2_data[vindex].size);
        _file.seekg(_opt2_data[vindex].offset, std::ios::beg);
        _file.read(reinterpret_cast<char *>(opt_vars[1]->buffer()), opt_vars[1]->total_size());

        vindex++;
      });
  }

private:
  std::ifstream _file;
  checkpoint::Header _header;
  checkpoint::Footer _footer;
  DataBuffer _tensor_data;
  DataBuffer _opt1_data;
  DataBuffer _opt2_data;
};

void loadCheckpoint(const std::string &filename,
                    const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                    const std::unique_ptr<onert::exec::Execution> &exec)
{
  CheckpointLoader loader(filename);
  loader.updateTensor(exec);
  loader.updateOptimizer(train_info, exec);
  loader.updateTrainingInfo(train_info);
}

} // namespace loader
} // namespace onert
