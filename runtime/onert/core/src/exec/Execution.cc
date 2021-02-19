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

#include "util/logging.h"

namespace onert
{
namespace exec
{

Execution::Execution(const std::shared_ptr<ExecutorMap> &executors) : _executors{executors}
{
  assert(executors != nullptr);
  assert(executors->at(ir::SubgraphIndex{0}) != nullptr);
  const auto &primary_subg = primary_subgraph();
  _io_desc.inputs.resize(primary_subg.getInputs().size());
  _io_desc.outputs.resize(primary_subg.getOutputs().size());

  sem_init(&_async_finish, 0, 0);
  sem_init(&_deque, 0, 1);
  sem_init(&_async_input_sem, 0, 0);
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
  const auto input_index = primary_subgraph().getInputs().at(index);
  const auto info = primary_subgraph().operands().at(input_index).info();

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

void Execution::CreateNewAsyncDesc()
{
  IODescription *_async_io_desc = new IODescription;
  _async_io_desc->inputs.resize(primary_subgraph().getInputs().size());
  _async_io_desc->outputs.resize(primary_subgraph().getOutputs().size());

  deque_wait();
  _async_io_descs.push_back(_async_io_desc);
  deque_post();
}

void Execution::finish_post()
{
  sem_post(&_async_finish);
}

void Execution::finish_wait()
{
  sem_wait(&_async_finish);
}

void Execution::deque_post()
{
  sem_post(&_deque);
}

void Execution::deque_wait()
{
  sem_wait(&_deque);
}

void Execution::input_post()
{
  sem_post(&_async_input_sem);
}

void Execution::input_wait()
{
  sem_wait(&_async_input_sem);
}

void Execution::set_finish()
{
  finished = true;
}

IODescription* Execution::get_async_io_desc()
{
  deque_wait();
  IODescription* ret = _async_io_descs.front();
  _async_io_descs.pop_front();
  deque_post();

  return ret;
}

void Execution::push_async_result(IODescription* io_desc)
{
  _async_result.push_back(io_desc);
}

bool Execution::is_empty_queue()
{
  return _async_io_descs.empty();
}

void Execution::get_result(std::vector<void *> outputs)
{
  IODescription* ret = _async_result.front();
  _async_result.pop_front();
  for (uint i = 0; i < ret->outputs.size(); i++) {
    memcpy(outputs[i], ret->outputs[i]->buffer, ret->outputs[i]->size);
  }
  free(ret);
}

void Execution::setAsyncInput(const ir::IOIndex &index, const void *buffer, size_t length,
                              ir::Layout layout)
{
  deque_wait();
  const auto input_index = primary_subgraph().getInputs().at(index);
  const auto info = primary_subgraph().operands().at(input_index).info();
  IODescription *_async_io_desc = _async_io_descs.back();

  {
    auto input_shape_sig = _async_io_desc->dynamic_input_shapes.find(index);
    auto size_required =
      (input_shape_sig != _async_io_desc->dynamic_input_shapes.end())
        ? input_shape_sig->second.num_elements() * onert::ir::sizeOfDataType(info.typeInfo().type())
        : info.total_size();

    if (length < size_required)
    {
      throw std::runtime_error{"Too small length"};
    }
  }
  void *_buffer = (void *)malloc(sizeof(void *) * length);
  memcpy(_buffer, buffer, length);

  _async_io_desc->inputs.at(index.value()) = std::make_unique<InputDesc>(info, _buffer, length, layout);
  deque_post();
}

void Execution::setAsyncOutput(const ir::IOIndex &index, void *buffer, size_t length,
                               ir::Layout layout)
{
  deque_wait();
  const auto output_index = primary_subgraph().getOutputs().at(index);
  const auto info = primary_subgraph().operands().at(output_index).info();
  IODescription *_async_io_desc = _async_io_descs.front();

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _async_io_desc->outputs.at(index.value()) = std::make_unique<OutputDesc>(info, buffer, length, layout);
  deque_post();
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
  const auto output_index = primary_subgraph().getOutputs().at(index);
  const auto info = primary_subgraph().operands().at(output_index).info();

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

  primary_executor()->execute(_io_desc);
  finished = true;

  VERBOSE(Execution) << "Execution finished" << std::endl;
}

void Execution::Async_execute()
{
  VERBOSE(Execution) << "Start Async execution" << std::endl;
  if (_async_io_descs.empty()) {
    VERBOSE(Execution) << "The input is not ready" << std::endl;
    return;
  }

  deque_wait();
  primary_executor()->execute(*_async_io_descs.front());
  deque_post();
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

ir::Shape Execution::getInputShape(ir::IOIndex ind) const
{
  auto itr = _io_desc.dynamic_input_shapes.find(ind);
  if (itr == _io_desc.dynamic_input_shapes.end())
  {
    auto operand_idx = primary_subgraph().getInputs().at(ind);
    return primary_subgraph().operands().at(operand_idx).shape();
  }
  else
  {
    return itr->second;
  }
}

ir::Shape Execution::getOutputShape(ir::IOIndex ind) const
{
  if (!isFinished())
    throw std::runtime_error("Cannot get output shape before execution is finished");

  const auto &output_desc = _io_desc.outputs.at(ind.value());

  return output_desc->info.shape();
}

} // namespace exec
} // namespace onert
