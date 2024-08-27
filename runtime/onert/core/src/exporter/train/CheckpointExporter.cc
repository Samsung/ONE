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

class CheckpointExporter
{
public:
  CheckpointExporter(const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                     const std::unique_ptr<exec::Execution> &exec)
  {
    std::memset(&_header, 0, sizeof(_header));
    _header.magic = checkpoint::MAGIC_NUMBER;
    _header.schema = checkpoint::SCHEMA_VERSION;

    uint32_t offset = sizeof(_header);
    // TODO Store tensor and optimizer data
    UNUSED_RELEASE(exec);
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
    // TODO Write tensor and optimizer data
    dst.write(reinterpret_cast<const char *>(&_footer), sizeof(_footer));
    dst.close();
  }

private:
  checkpoint::Header _header;
  checkpoint::Footer _footer;
};

} // namespace

namespace onert
{
namespace exporter
{
namespace train
{

void exportCheckpoint(const std::string &filename,
                      const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                      const std::unique_ptr<exec::Execution> &exec)
{
  CheckpointExporter exporter(train_info, exec);
  exporter.save(filename);
}

} // namespace train
} // namespace exporter
} // namespace onert
