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
  sem_init(&_async_io_descs_sem, 0, 1);
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

void Execution::createNewAsyncDesc(uint32_t count)
{
  IODescription *_async_io_desc = new IODescription;
  _async_io_desc->inputs.resize(primary_subgraph().getInputs().size());
  _async_io_desc->outputs.resize(primary_subgraph().getOutputs().size());

  _async_io_descs.push_back({_async_io_desc, count});
}

void Execution::setFinish() { finished = true; }

bool Execution::isEmptyQueue()
{
  asyncIoDescSemWait();
  bool ret = _async_io_descs.empty();
  if (!ret)
  {
    for (uint32_t idx = 0; idx < _async_io_descs.front().first->inputs.size(); idx++)
    {
      if (_async_io_descs.front().first->inputs.at(idx).get() == nullptr)
      {
        ret = true;
        break;
      }
    }
  }
  asyncIoDescSemPost();
  return ret;
}

void Execution::executeAsyncInput(const ir::IOIndex &index, const void *buffer, size_t length,
                                  ir::Layout layout)
{
  const auto input_index = primary_subgraph().getInputs().at(index);
  const auto info = primary_subgraph().operands().at(input_index).info();
  IODescription *_async_io_desc = _async_io_descs.back().first;

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
  void *_buffer = (void *)malloc(length);
  if (_buffer == NULL)
  {
    throw std::runtime_error{"malloc failed"};
  }
  memcpy(_buffer, buffer, length);

  _async_io_desc->inputs.at(index.value()) =
    std::make_unique<InputDesc>(info, _buffer, length, layout);
}

void Execution::executeAsyncOutput(const ir::IOIndex &index, void *buffer, size_t length,
                                   ir::Layout layout)
{
  const auto output_index = primary_subgraph().getOutputs().at(index);
  const auto info = primary_subgraph().operands().at(output_index).info();
  IODescription *_async_io_desc = _async_io_descs.front().first;

  if (length < info.total_size())
  {
    throw std::runtime_error{"Too small length"};
  }

  _async_io_desc->outputs.at(index.value()) =
    std::make_unique<OutputDesc>(info, buffer, length, layout);
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

void Execution::AsyncExecute()
{
  VERBOSE(Execution) << "Start Async execution" << std::endl;
  if (_async_io_descs.empty())
  {
    VERBOSE(Execution) << "The input is not ready" << std::endl;
    return;
  }

  primary_executor()->execute(*_async_io_descs.front().first);
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

void Execution::asyncIoDescSemWait() { sem_wait(&_async_io_descs_sem); }

void Execution::asyncIoDescSemPost() { sem_post(&_async_io_descs_sem); }

void Execution::runInference()
{
  uint32_t inference_cnt;
  uint32_t output_sz = primary_subgraph().getOutputs().size();
  while (true)
  {
    if (isEmptyQueue())
    {
      if (isFinished())
      {
        if (!next_exes.empty())
        {
          for (uint32_t i = 0; i < next_exes.size(); i++)
          {
            std::get<0>(next_exes[i])->setFinish();
          }
        }
        else
        {
          sholudStop();
        }
        break;
      }
    }
    else
    {
      for (uint32_t i = 0; i < output_sz; i++)
      {
        auto opidx = primary_subgraph().getOutputs().at(i);
        auto shape = primary_subgraph().operands().at(opidx).shape();
        auto dtype = primary_subgraph().operands().at(opidx).typeInfo().type();
        auto rank = shape.rank();
        uint32_t tensor_size = 1;
        for (int32_t j = 0; j < rank; j++)
        {
          tensor_size *= shape.dim(j);
        }
        if (dtype == onert::ir::DataType::FLOAT32 || dtype == onert::ir::DataType::INT32 ||
            dtype == onert::ir::DataType::UINT32)
          tensor_size *= 4;
        else if (dtype == onert::ir::DataType::INT64)
          tensor_size *= 8;
        void *_buffer = (void *)malloc(tensor_size);
        if (_buffer == NULL)
        {
          throw std::runtime_error{"malloc failed"};
        }
        executeAsyncOutput(onert::ir::IOIndex(i), _buffer, tensor_size);
      }
      AsyncExecute();

      // set inputs of next execution
      auto _io_desc = getAsyncIoDescs()->front().first;
      inference_cnt = getAsyncIoDescs()->front().second;
      getAsyncIoDescs()->pop_front();

      for (uint32_t i = 0; i < next_exes.size(); i++)
      {
        auto next_exe = std::get<0>(next_exes[i]);
        auto o_index = std::get<1>(next_exes[i]);
        auto i_index = std::get<2>(next_exes[i]);

        next_exe->asyncIoDescSemWait();
        auto next_io_descs = next_exe->getAsyncIoDescs();
        bool exist = false;
        for (auto iter = next_io_descs->begin(); iter != next_io_descs->end(); iter++)
        {
          if (inference_cnt == iter->second)
          {
            exist = true;
          }
        }

        if (!exist)
        {
          next_exe->createNewAsyncDesc(inference_cnt);
        }
        for (auto iter = next_io_descs->begin(); iter != next_io_descs->end(); iter++)
        {
          if (inference_cnt == iter->second)
          {
            const auto input_index = next_exe->primary_subgraph().getInputs().at(i_index.value());
            const auto info = next_exe->primary_subgraph().operands().at(input_index).info();

            size_t length = _io_desc->outputs[o_index.value()]->size;
            void *_buffer = (void *)malloc(length);
            if (_buffer == NULL)
            {
              throw std::runtime_error{"malloc failed"};
            }
            memcpy(_buffer, _io_desc->outputs[o_index.value()]->buffer, length);

            iter->first->inputs.at(i_index.value()) = std::make_unique<onert::exec::InputDesc>(
              info, _buffer, length, onert::ir::Layout::NHWC);
            break;
          }
        }
        next_exe->asyncIoDescSemPost();
      }

      if (next_exes.empty())
      {
        std::vector<void *> results;
        for (uint32_t i = 0; i < _io_desc->outputs.size(); i++)
        {
          size_t length = _io_desc->outputs[i]->size;
          void *_buffer = (void *)malloc(length);
          if (_buffer == NULL)
          {
            throw std::runtime_error{"malloc failed"};
          }
          memcpy(_buffer, _io_desc->outputs[i]->buffer, length);
          results.push_back(_buffer);
        }
        _async_results.push_back(results);
      }

      for (uint32_t i = 0; i < _io_desc->inputs.size(); i++)
      {
        auto p = _io_desc->inputs.at(i).release();
        if (p)
        {
          free((void *)p->buffer);
          delete p;
        }
      }
      for (uint32_t i = 0; i < _io_desc->outputs.size(); i++)
      {
        auto p = _io_desc->outputs.at(i).release();
        if (p)
        {
          free(p->buffer);
          delete p;
        }
      }
      delete _io_desc;
    }
  }
}

bool Execution::stopWait(void) const { return stop_wait; }

void Execution::sholudStop() { stop_wait = true; }

} // namespace exec
} // namespace onert
