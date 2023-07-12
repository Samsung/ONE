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

#include "train/TrainableExecutors.h"

#include "util/logging.h"

namespace onert
{
namespace exec
{

Execution::Execution(const std::shared_ptr<IExecutors> &executors) : _executors{executors}
{
  assert(executors != nullptr);
  assert(executors->entryExecutor() != nullptr);
  _io_desc.inputs.resize(_executors->inputSize());
  _io_desc.outputs.resize(_executors->outputSize());
}

void Execution::changeInputShape(const ir::IOIndex &index, const ir::Shape &new_shape)
{
  // This will be used later to set input tensor dynamic
  // Note that 'compiled' model will not be updated with new_shape
  // but new_shape will change model input shape while 'running' the model
  _io_desc.dynamic_input_shapes[index] = new_shape;

  VERBOSE(Execution) << "Model input shape will be changed at the start of execute()"
                     << "(index: " << index << ")" << std::endl;
}

// TODO Remove default parameter
void Execution::setInput(const ir::IOIndex &index, const void *buffer, size_t length,
                         ir::Layout layout)
{
  const auto info = _executors->inputInfo(index);

  // TODO handle when (!buffer && length != 0) : setting the input as an optional tensor

  // check if size enough for input is passed
  // if input_shape_sig is set, input_shape_sig overrides shape in info
  // note: input_shape_sig contains shape passed by nnfw_set_input_tensorinfo()
  {
    auto input_shape_sig = _io_desc.dynamic_input_shapes.find(index);
    auto size_required =
      (input_shape_sig != _io_desc.dynamic_input_shapes.end())
        ? input_shape_sig->second.num_elements() * onert::ir::sizeOfDataType(info.typeInfo().type())
        : info.total_size();

    if (length < size_required)
    {
      throw std::runtime_error{"Too small length"};
    }
  }

  _io_desc.inputs.at(index.value()) = std::make_unique<InputDesc>(info, buffer, length, layout);
}

// TODO Remove default parameter
void Execution::setInput(const ir::IOIndex &index, const ir::TypeInfo &type, const ir::Shape &shape,
                         const void *buffer, size_t length, ir::Layout layout)
{
  auto info = ir::OperandInfo::createStaticInfo(shape, type);

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _io_desc.inputs.at(index.value()) = std::make_unique<InputDesc>(info, buffer, length, layout);
}

// TODO Remove default parameter
void Execution::setOutput(const ir::IOIndex &index, void *buffer, size_t length, ir::Layout layout)
{
  const auto info = _executors->outputInfo(index);

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _io_desc.outputs.at(index.value()) = std::make_unique<OutputDesc>(info, buffer, length, layout);
}

// TODO Remove default parameter
void Execution::setOutput(const ir::IOIndex &index, const ir::TypeInfo &type,
                          const ir::Shape &shape, void *buffer, size_t length, ir::Layout layout)
{
  auto info = ir::OperandInfo::createStaticInfo(shape, type);

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _io_desc.outputs.at(index.value()) = std::make_unique<OutputDesc>(info, buffer, length, layout);
}

void Execution::setInputLayout(const ir::IOIndex &index, ir::Layout layout)
{
  const auto &input_desc = _io_desc.inputs.at(index.value());
  _io_desc.inputs.at(index.value()) =
    std::make_unique<InputDesc>(input_desc->info, input_desc->buffer, input_desc->size, layout);
}

void Execution::setOutputLayout(const ir::IOIndex &index, ir::Layout layout)
{
  const auto &output_desc = _io_desc.outputs.at(index.value());
  _io_desc.outputs.at(index.value()) =
    std::make_unique<OutputDesc>(output_desc->info, output_desc->buffer, output_desc->size, layout);
}

void Execution::execute()
{
  VERBOSE(Execution) << "Start execution" << std::endl;

  _executors->execute(_io_desc);
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

#ifdef ONERT_TRAIN
void Execution::train()
{
  auto execs = dynamic_cast<exec::train::TrainableExecutors *>(_executors.get());
  if (!execs)
  {
    throw std::runtime_error{"Supported only TrainableExecutors"};
  }

  VERBOSE(Execution) << "Start training" << std::endl;

  execs->train(_io_desc);
  finished = true;

  VERBOSE(Execution) << "training finished" << std::endl;
}
#endif // ONERT_TRAIN

ir::Shape Execution::getInputShape(ir::IOIndex ind) const
{
  auto itr = _io_desc.dynamic_input_shapes.find(ind);
  if (itr == _io_desc.dynamic_input_shapes.end())
  {
    return _executors->inputInfo(ind).shape();
  }
  else
  {
    return itr->second;
  }
}

// NNAPI return fail if ANeuralNetworksExecution_getOutputOperandRank or
// ANeuralNetworksExecution_getOutputOperandDimensions is called before execution.
// On the other hand, NNFW API return static shape inference result if nnfw_output_tensorinfo is
// called before execution.
// To handle both case, this method retun static shape inference result and fail will be handled on
// NNAPI frontend.
ir::Shape Execution::getOutputShape(ir::IOIndex ind) const
{
  if (!isFinished())
    return _executors->outputInfo(ind).shape();

  const auto &output_desc = _io_desc.outputs.at(ind.value());

  return output_desc->info.shape();
}

} // namespace exec
} // namespace onert
