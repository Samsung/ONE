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
#include <array>

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

void SingleModelExecutors::execute(const ExecutionContext &ctx)
{
  // Create Input/Output UserTensors
  std::vector<std::unique_ptr<backend::builtin::UserTensor>> tensorpool;
  std::vector<backend::IPortableTensor *> inputs(ctx.desc.inputs.size());
  std::vector<backend::IPortableTensor *> outputs(ctx.desc.outputs.size());

  for (uint32_t i = 0; i < inputs.size(); i++)
  {
    auto &desc = ctx.desc.inputs[i];

    // Input is optional if buffer is nullptr, and optional input's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Input " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, static_cast<const uint8_t *>(desc->buffer), desc->size));
    inputs[i] = tensorpool.back().get();
  }
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    auto &desc = ctx.desc.outputs[i];

    // Output is optional if buffer is nullptr, and optional output's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, static_cast<const uint8_t *>(desc->buffer), desc->size));
    outputs[i] = tensorpool.back().get();
  }

  entryExecutor()->execute(inputs, outputs, ctx.options);

  // Get dynamic shape inference result
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    if (ctx.desc.outputs[i]->buffer == nullptr)
    {
      // Output is optional if buffer is nullptr
      continue;
    }

    ctx.desc.outputs[i]->info.shape(outputs[i]->getShape());
  }
}

} // namespace exec
} // namespace onert
