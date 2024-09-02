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

#ifndef __ONERT_EXPORTER_TRAIN_CHECKPOINT_EXPORTER_H__
#define __ONERT_EXPORTER_TRAIN_CHECKPOINT_EXPORTER_H__

#include <string>
#include <vector>
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
} // namespace onert

namespace onert
{
namespace exporter
{
namespace train
{

void exportCheckpoint(const std::string &filename,
                      const std::unique_ptr<ir::train::TrainingInfo> &train_info,
                      const std::unique_ptr<exec::Execution> &exec);

} // namespace train
} // namespace exporter
} // namespace onert

#endif // __ONERT_EXPORTER_TRAIN_CHECKPOINT_EXPORTER_H__
