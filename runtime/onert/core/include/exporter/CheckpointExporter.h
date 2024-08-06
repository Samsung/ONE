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
                     std::unique_ptr<onert::exec::Execution> &exec);

  void save(const std::string &path);

private:
  void setReservedData();
  void setTensorData(std::unique_ptr<onert::exec::Execution> &exec);
  void setOptimizerData(std::unique_ptr<onert::ir::train::TrainingInfo> &train_info,
                        std::unique_ptr<onert::exec::Execution> &exec);
  void setAdamOptimizerData(std::unique_ptr<onert::exec::Execution> &exec);

private:
  const uint32_t RESERVED_SIZE = 16;

  std::vector<char> _reserved;
  std::vector<char> _buffers;
  std::vector<char> _optimizers;
  std::mutex _mutex;
};

} // namespace exporter
} // namespace onert
