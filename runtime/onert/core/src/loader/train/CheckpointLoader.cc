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

#include <fstream>
#include <filesystem>

namespace
{

using namespace onert;
using namespace ir;
using namespace train;
using namespace checkpoint;
using namespace exec;

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

private:
  std::ifstream _file;
  checkpoint::Header _header;
  checkpoint::Footer _footer;
};

} // namespace

namespace onert
{
namespace loader
{
namespace train
{

void loadCheckpoint(const std::string &filename,
                    const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                    const std::unique_ptr<onert::exec::Execution> &exec)
{
  CheckpointLoader loader(filename);

  // TODO Load tensor data
  UNUSED_RELEASE(exec);
  // TODO Update step in train_info
  UNUSED_RELEASE(train_info);
}

} // namespace train
} // namespace loader
} // namespace onert
