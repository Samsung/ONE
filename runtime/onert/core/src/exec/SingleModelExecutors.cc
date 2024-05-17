/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SingleModelExecutors.h"

#include "../backend/builtin/IOTensor.h"

namespace onert
{
namespace exec
{

void SingleModelExecutors::emplace(const ir::ModelIndex &, const ir::SubgraphIndex &subg_index,
                                   std::unique_ptr<IExecutor> exec)
{
  _executors.emplace(subg_index, std::move(exec));
}

IExecutor *SingleModelExecutors::at(const ir::ModelIndex &,
                                    const ir::SubgraphIndex &subg_index) const
{
  return _executors.at(subg_index).get();
}

uint32_t SingleModelExecutors::inputSize() const
{
  return entryExecutor()->getInputTensors().size();
}

uint32_t SingleModelExecutors::outputSize() const
{
  return entryExecutor()->getOutputTensors().size();
}

const ir::OperandInfo &SingleModelExecutors::inputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->getInputTensors().at(index.value())->orig_info();
}

const ir::OperandInfo &SingleModelExecutors::outputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->getOutputTensors().at(index.value())->orig_info();
}

void SingleModelExecutors::execute(const ExecutionContext &ctx) { entryExecutor()->execute(ctx); }

} // namespace exec
} // namespace onert
