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

#include "TrainableExecutors.h"

#include "../../backend/builtin/IOTensor.h"

#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace exec
{
namespace train
{

void TrainableExecutors::emplace(const ir::ModelIndex &, const ir::SubgraphIndex &subg_index,
                                 std::unique_ptr<IExecutor> exec)
{
  std::unique_ptr<TrainableExecutor> t_exec{
    nnfw::misc::polymorphic_downcast<TrainableExecutor *>(exec.release())};
  _executors.emplace(subg_index, std::move(t_exec));
}

TrainableExecutor *TrainableExecutors::at(const ir::ModelIndex &,
                                          const ir::SubgraphIndex &subg_index) const
{
  return _executors.at(subg_index).get();
}

uint32_t TrainableExecutors::inputSize() const { return entryExecutor()->getInputTensors().size(); }

uint32_t TrainableExecutors::outputSize() const
{
  return entryExecutor()->getOutputTensors().size();
}

const ir::OperandInfo &TrainableExecutors::inputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->getInputTensors().at(index.value())->orig_info();
}

const ir::OperandInfo &TrainableExecutors::outputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->getOutputTensors().at(index.value())->orig_info();
}

void TrainableExecutors::execute(const IODescription &desc)
{
  if (_executors.size() > 1)
    throw std::runtime_error("TrainableExecutors does not support multiple executors yet");
  entryExecutor()->execute(desc);

  // TODO Support multple executors
}

void TrainableExecutors::train(const IODescription &desc)
{
  if (_executors.size() > 1)
    throw std::runtime_error("TrainableExecutors does not support multiple executors yet");
  entryExecutor()->train(desc);

  // TODO Support multple executors
}

} // namespace train
} // namespace exec
} // namespace onert
