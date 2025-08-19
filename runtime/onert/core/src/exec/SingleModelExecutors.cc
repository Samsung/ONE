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

#include "EdgeTensor.h"
#include "IPermuteFunction.h"
#include "../backend/builtin/UserTensor.h"
#include "../backend/builtin/IOTensor.h"

namespace onert::exec
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

uint32_t SingleModelExecutors::inputSize() const { return entryExecutor()->inputSize(); }

uint32_t SingleModelExecutors::outputSize() const { return entryExecutor()->outputSize(); }

const ir::OperandInfo &SingleModelExecutors::inputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->inputInfo(index.value());
}

const ir::OperandInfo &SingleModelExecutors::outputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->outputInfo(index.value());
}

const void *SingleModelExecutors::outputBuffer(const ir::IOIndex &index) const
{
  return static_cast<const void *>(entryExecutor()->outputBuffer(index.value()));
}

const backend::IPortableTensor *SingleModelExecutors::outputTensor(const ir::IOIndex &index) const
{
  return entryExecutor()->outputTensor(index.value());
}

void SingleModelExecutors::execute(ExecutionContext &ctx)
{
  // UserTensor for Input/Output
  std::vector<std::unique_ptr<backend::builtin::UserTensor>> tensorpool;

  // Input/Output Tensor vector for executor
  std::vector<backend::IPortableTensor *> inputs(ctx.desc.inputs.size());
  std::vector<backend::IPortableTensor *> outputs(ctx.desc.outputs.size());

  // Prepare UserTensor for input
  for (uint32_t i = 0; i < inputs.size(); i++)
  {
    auto &desc = ctx.desc.inputs[i];

    // Input is optional if buffer is nullptr, and optional input's size is 0
    if (desc.buffer == nullptr && (desc.size != 0 || desc.info.total_size() != 0))
      throw std::runtime_error{"Input " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc.info, const_cast<uint8_t *>(static_cast<const uint8_t *>(desc.buffer)), desc.size));

    inputs[i] = tensorpool.back().get();
  }

  // Prepare UserTensor for output
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    auto &desc = ctx.desc.outputs[i];
    const auto output_io_tensor =
      dynamic_cast<const backend::builtin::IOTensor *>(outputTensor(ir::IOIndex{i}));
    if (!output_io_tensor)
      throw std::runtime_error{"Output tensor must be IOTensor"};
    bool skip_set_output = output_io_tensor->hasBackendTensor();

    // If buffer is nullptr, output is optional or internally allocated buffer,
    // and optional output's size is 0
    if (desc.buffer == nullptr && (desc.size != 0 || desc.info.total_size() != 0) &&
        !skip_set_output)
      throw std::runtime_error{"Output " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc.info, static_cast<uint8_t *>(desc.buffer), desc.size));
    outputs[i] = tensorpool.back().get();
  }

  // Executor
  entryExecutor()->execute(inputs, outputs, ctx.options);

  // Get dynamic shape inference result
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    const auto output_io_tensor = outputTensor(ir::IOIndex{i});
    ctx.desc.outputs[i].info.shape(output_io_tensor->get_info().shape());
  }
}

} // namespace onert::exec
