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

namespace onert::exec::train
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

uint32_t TrainableExecutors::inputSize() const { return entryExecutor()->inputSize(); }

uint32_t TrainableExecutors::outputSize() const { return entryExecutor()->outputSize(); }

const ir::OperandInfo &TrainableExecutors::inputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->inputInfo(index.value());
}

const ir::OperandInfo &TrainableExecutors::outputInfo(const ir::IOIndex &index) const
{
  return entryExecutor()->outputInfo(index.value());
}

const void *TrainableExecutors::outputBuffer(const ir::IOIndex &index) const
{
  return static_cast<const void *>(entryExecutor()->outputBuffer(index.value()));
}

const backend::IPortableTensor *TrainableExecutors::outputTensor(const ir::IOIndex &index) const
{
  return entryExecutor()->outputTensor(index.value());
}

void TrainableExecutors::execute(const ExecutionContext &ctx)
{
  if (_executors.size() > 1)
    throw std::runtime_error("TrainableExecutors does not support multiple executors yet");

  // UserTensor for Input/Output
  std::vector<std::unique_ptr<backend::builtin::UserTensor>> tensorpool;

  // Allocate UserTensor and call executor forward
  forward(ctx, tensorpool, false);

  // TODO Support multple executors
}

void TrainableExecutors::train(const ExecutionContext &ctx, uint32_t training_step)
{
  if (_executors.size() > 1)
    throw std::runtime_error("TrainableExecutors does not support multiple executors yet");

  // UserTensor for Input/Output
  std::vector<std::unique_ptr<backend::builtin::UserTensor>> tensorpool;

  // Allocate UserTensor and call executor forward and backward
  forward(ctx, tensorpool, true);
  entryExecutor()->backward(ctx.options, training_step);

  // TODO Support multple executors
}

void TrainableExecutors::forward(
  const ExecutionContext &ctx,
  std::vector<std::unique_ptr<backend::builtin::UserTensor>> &tensorpool, bool training)
{
  // Input/Output Tensor vector for executor
  std::vector<backend::IPortableTensor *> inputs(ctx.desc.inputs.size());
  std::vector<backend::IPortableTensor *> outputs(ctx.desc.outputs.size());

  // Prepare UserTensor for input
  for (uint32_t i = 0; i < inputs.size(); i++)
  {
    auto &desc = ctx.desc.inputs[i];

    // Input is optional if buffer is nullptr, and optional input's size is 0
    if (desc->buffer == nullptr && (desc->size != 0 || desc->info.total_size() != 0))
      throw std::runtime_error{"Input " + std::to_string(i) + "'s buffer is not set."};

    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, const_cast<uint8_t *>(static_cast<const uint8_t *>(desc->buffer)),
      desc->size));
    inputs[i] = tensorpool.back().get();
  }

  // Prepare UserTensor for output
  for (uint32_t i = 0; i < outputs.size(); i++)
  {
    auto &desc = ctx.desc.outputs[i];

    // If training, output buffer may not be used
    // So don't check optional
    tensorpool.emplace_back(std::make_unique<backend::builtin::UserTensor>(
      desc->info, desc->layout, static_cast<uint8_t *>(desc->buffer), desc->size));
    outputs[i] = tensorpool.back().get();
  }

  // Call forward
  entryExecutor()->forward(inputs, outputs, ctx.options, training);
}

float TrainableExecutors::getLoss(const ir::IOIndex &index) const
{
  if (_executors.size() > 1)
    throw std::runtime_error("TrainableExecutors does not support multiple executors yet");
  return entryExecutor()->getLoss(index);
}

void TrainableExecutors::iterateTrainableTensors(
  const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)> &fn)
  const
{
  return entryExecutor()->iterateTrainableTensors(fn);
}

} // namespace onert::exec::train
