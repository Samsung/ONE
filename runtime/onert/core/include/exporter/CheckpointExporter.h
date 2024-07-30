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

#include <string>
#include <vector>
#include <memory>
#include <mutex>

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
class CheckpointExporter
{
public:
  CheckpointExporter(std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
                     std::unique_ptr<onert::exec::Execution> &execution);

  void save(const std::string &path);

private:
  uint32_t getTotalSize();

private:
  const uint16_t MAGIC_NUMBER = 429;
  const uint8_t SCHEMA_VERSION = 1;
  const uint8_t RESERVED = 0;

  std::vector<char> _data;
  std::mutex _mutex;
};

} // namespace exporter
} // namespace onert
