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

#include "circle_schema_generated.h"

#include <memory>
#include <string>
#include <memory>

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
namespace exporter
{
class MMappedFile;
} // namespace exporter
} // namespace onert

namespace onert
{
namespace exporter
{
class CircleExporter
{
public:
  CircleExporter(const std::string &source, const std::string &path);
  ~CircleExporter();

  void updateWeight(const std::unique_ptr<onert::exec::Execution> &exec);
  void updateMetadata(const std::unique_ptr<onert::ir::train::TrainingInfo> &training_info);
  void finish();

private:
  std::string _path;
  std::string _data;
  std::unique_ptr<::circle::ModelT> _model;
};
} // namespace exporter
} // namespace onert
