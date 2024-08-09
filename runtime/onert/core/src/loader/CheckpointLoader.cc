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
#include "ir/train/TrainingInfo.h"
#include "exec/Execution.h"
#include "util/Utils.h"

#include <fstream>
#include <filesystem>
// #include "BaseLoader.h"
// #include "circle_schema_generated.h"

namespace onert
{
namespace loader
{

namespace checkpoint
{

struct __attribute__((packed)) Header
{
  uint16_t magic;
  uint8_t schema;
  uint8_t reserved;
  uint32_t opt1_offset;
  uint32_t opt2_offset;
  uint32_t other_offset;
  uint32_t length;
};

} // namespace checkpoint

struct DataBufferPair
{
  DataBufferPair(uint32_t _offset, uint32_t _size) : offset{_offset}, size{_size}
  {
    // DO NOTHING
  }

  uint32_t offset;
  uint32_t size;
};

struct DataBuffer
{
  std::vector<uint32_t> offset;
  std::vector<uint32_t> size;

  void resize(uint32_t length)
  {
    offset.resize(length);
    size.resize(length);
  }

  char *getOffsetBuf() { return reinterpret_cast<char *>(offset.data()); }

  void calculateSize(uint32_t next_beg_offset)
  {
    assert(offset.size() == size.size());
    uint32_t cur = offset[0];
    for (size_t i = 1; i < offset.size(); ++i)
    {
      size[i - 1] = offset[i] - cur;
      cur = offset[i];
    }
    size.back() = next_beg_offset - offset.back();
  }

  // offset, size
  DataBufferPair operator[](uint32_t i)
  {
    assert(offset.size() == size.size());
    assert(i <= offset.size());
    return DataBufferPair{offset[i], size[i]};
  }
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

    if (_header.magic != MAGIC_NUMBER)
      throw std::runtime_error{"Invalid MAGIC NUMBER"};

    if (_header.schema != SCHEMA_VERSION)
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
      [&](const ir::OperandIndex &, const backend::train::ITrainableTensor *tensor) {
        assert(tensor);
        assert(tensor->total_size() == _tensor_data[vindex].size);
        _file.seekg(_tensor_data[vindex].offset, std::ios::beg);
        _file.read(reinterpret_cast<char *>(tensor->buffer()), tensor->total_size());
        vindex++;
      });
  }

private:
  static constexpr uint16_t MAGIC_NUMBER = 429;
  static constexpr uint8_t SCHEMA_VERSION = 1;

  std::ifstream _file;
  checkpoint::Header _header;
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
  // loader.updateOptimizer(train_info, exec);
  UNUSED_RELEASE(train_info);
}

} // namespace loader
} // namespace onert
