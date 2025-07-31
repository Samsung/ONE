/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "exec/Execution.h"

#include "../backend/builtin/IOTensor.h"
#include "ir/DataType.h"
#include "train/TrainableExecutors.h"
#include "util/logging.h"

#include <memory>

namespace onert::exec
{

Execution::Execution(const std::shared_ptr<IExecutors> &executors) : _executors{executors}
{
  assert(executors != nullptr);
  assert(executors->entryExecutor() != nullptr);

  // Initialize I/O description
  _ctx.desc.inputs.resize(_executors->inputSize());
  for (uint32_t i = 0; i < _executors->inputSize(); ++i)
    _ctx.desc.inputs.at(i) = std::make_unique<InputDesc>(_executors->inputInfo(ir::IOIndex(i)));

  _ctx.desc.outputs.resize(_executors->outputSize());
  for (uint32_t i = 0; i < _executors->outputSize(); ++i)
    _ctx.desc.outputs.at(i) = std::make_unique<OutputDesc>(_executors->outputInfo(ir::IOIndex(i)));
  _ctx.shape_updated = false;

  _is_internal_output_tensor.resize(_executors->outputSize());
  for (uint32_t i = 0; i < _executors->outputSize(); ++i)
  {
    const auto output_tensor =
      dynamic_cast<const backend::builtin::IOTensor *>(executors->outputTensor(ir::IOIndex{i}));
    if (!output_tensor)
      throw std::runtime_error("Output tensor must be IOTensor");

    _is_internal_output_tensor.at(i) = output_tensor->hasBackendTensor();
  }

  // Initialize options
  ExecutionOptions::fromGlobalConfig(_ctx.options);
}

void Execution::changeInputShape(const ir::IOIndex &index, const ir::Shape &new_shape)
{
  // This will be used later to set input tensor dynamic
  // Note that 'compiled' model will not be updated with new_shape
  // but new_shape will change model input shape while 'running' the model
  auto &input_desc = _ctx.desc.inputs.at(index.value());
  if (new_shape != input_desc->info.shape())
  {
    input_desc->info.shape(new_shape);
    _ctx.shape_updated = true;

    VERBOSE(Execution) << "Model input shape will be changed at the start of execute()"
                       << "(index: " << index << ")" << std::endl;
  }
}

// TODO Remove default parameter
void Execution::setInput(const ir::IOIndex &index, const void *buffer, size_t length)
{
  // Length validation in execute(): datatype can be changed by API call
  auto &input_desc = _ctx.desc.inputs.at(index.value());
  input_desc->buffer = buffer;
  input_desc->size = length;
}

void Execution::setInput(const ir::IOIndex &index, const ir::Shape &shape, const void *buffer,
                         size_t length)
{
  changeInputShape(index, shape);
  setInput(index, buffer, length);
}

void Execution::setOutput(const ir::IOIndex &index, void *buffer, size_t length)
{
  // Length validation in execute()
  // - datatype can be changed by API call
  // - shape can be changed by dynamic shape inference
  auto &output_desc = _ctx.desc.outputs.at(index.value());
  output_desc->buffer = buffer;
  output_desc->size = length;
}

void Execution::setOutput(const ir::IOIndex &index, const ir::Shape &shape, void *buffer,
                          size_t length)
{
  auto &output_desc = _ctx.desc.outputs.at(index.value());
  output_desc->info.shape(shape);

  setOutput(index, buffer, length);
}

void Execution::setInputLayout(const ir::IOIndex &index, ir::Layout layout)
{
  _ctx.desc.inputs.at(index.value())->layout = layout;
}

void Execution::setOutputLayout(const ir::IOIndex &index, ir::Layout layout)
{
  _ctx.desc.outputs.at(index.value())->layout = layout;
}

void Execution::execute()
{
  VERBOSE(Execution) << "Start execution" << std::endl;

  // Input length validation check
  for (const auto &input : _ctx.desc.inputs)
  {
    if (input->info.total_size() > input->size)
      throw std::runtime_error{"Too small input buffer length"};
  }

  // Output length validation check
  if (!_ctx.shape_updated)
  {
    assert(_ctx.desc.outputs.size() == _is_internal_output_tensor.size());
    for (uint32_t i = 0; i < _ctx.desc.outputs.size(); ++i)
    {
      const bool is_managed_internally = _is_internal_output_tensor.at(i);
      const auto &output = _ctx.desc.outputs.at(i);
      if (!is_managed_internally && output->info.total_size() > output->size)
        throw std::runtime_error{"Too small output buffer length"};
      if (is_managed_internally && output->buffer != nullptr)
        VERBOSE(Execution) << "Warning: Output buffer was set from API even though the output "
                              "tensor was allocated internally"
                           << std::endl;
    }
  }

  _executors->execute(_ctx);
  finished = true;

  VERBOSE(Execution) << "Execution finished" << std::endl;
}

void Execution::startExecute()
{
  VERBOSE(Execution) << "Create asynchronous execution thread" << std::endl;

  _exec_thread = std::make_unique<std::thread>(&Execution::execute, this);
}

void Execution::waitFinish()
{
  VERBOSE(Execution) << "Wait to finish execution" << std::endl;

  _exec_thread->join();
  finished = true;
}

bool Execution::isFinished(void) const { return finished; }

void Execution::train(uint32_t training_step)
{
  auto execs = dynamic_cast<exec::train::TrainableExecutors *>(_executors.get());
  if (!execs)
  {
    throw std::runtime_error{"Supported only TrainableExecutors"};
  }

  execs->train(_ctx, training_step);
  finished = true;
}

float Execution::getLoss(const ir::IOIndex &ind)
{
  auto execs = dynamic_cast<exec::train::TrainableExecutors *>(_executors.get());
  if (!execs)
  {
    throw std::runtime_error{"Supported only TrainableExecutors"};
  }

  return execs->getLoss(ind);
}

void Execution::iterateTrainableTensors(
  const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)> &fn)
  const
{
  auto execs = dynamic_cast<exec::train::TrainableExecutors *>(_executors.get());
  if (!execs)
  {
    throw std::runtime_error{"Supported only TrainableExecutors"};
  }
  execs->iterateTrainableTensors(fn);
}

ir::Shape Execution::getInputShape(ir::IOIndex ind) const
{
  return _ctx.desc.inputs.at(ind.value())->info.shape();
}

// NNAPI return fail if ANeuralNetworksExecution_getOutputOperandRank or
// ANeuralNetworksExecution_getOutputOperandDimensions is called before execution.
// On the other hand, NNFW API return static shape inference result if nnfw_output_tensorinfo is
// called before execution.
// To handle both case, this method retun static shape inference result and fail will be handled on
// NNAPI frontend.
ir::Shape Execution::getOutputShape(ir::IOIndex ind) const
{
  return _ctx.desc.outputs.at(ind.value())->info.shape();
}

size_t Execution::getInputTotalSize(ir::IOIndex ind) const
{
  // TODO Support dynamic shape
  return _ctx.desc.inputs.at(ind.value())->info.total_size();
}

size_t Execution::getOutputTotalSize(ir::IOIndex ind) const
{
  return _ctx.desc.outputs.at(ind.value())->info.total_size();
}

const void *Execution::getInputBuffer(ir::IOIndex ind) const
{
  return _ctx.desc.inputs.at(ind.value())->buffer;
}

void *Execution::getOutputBuffer(ir::IOIndex ind)
{
  return _ctx.desc.outputs.at(ind.value())->buffer;
}

} // namespace onert::exec
