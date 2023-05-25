/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_CORE_COMPILER_MANUAL_SCHEDULER_H__
#define __ONERT_CORE_COMPILER_MANUAL_SCHEDULER_H__

#include "IScheduler.h"
#include "compiler/Compiler.h"
#include "ir/train/TrainableGraph.h"

namespace onert
{
namespace compiler
{

class ManualScheduler : public IScheduler
{
public:
  ManualScheduler(const std::vector<const backend::Backend *> &backends,
                  const compiler::CompilerOptions &options);
  std::unique_ptr<BackendResolver> schedule(const ir::Graph &graph) override;

private:
  const backend::Backend *resolveBackend(const std::string &id,
                                         const backend::Backend *fallback = nullptr);

private:
  std::vector<const backend::Backend *> _backends;
  compiler::CompilerOptions _options;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_CORE_COMPILER_MANUAL_SCHEDULER_H__
