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

#include "nnfw_api_internal.h"

#include "util/Exceptions.h"
#include "util/logging.h"
#include "exec/Execution.h"

#include <iostream>
#include <string>
#include <vector>
#include <misc/string_helpers.h>

/*
 * API does not accept string argument longer than max length below
 */
#define MAX_BACKEND_NAME_LENGTH 32
#define MAX_OP_NAME_LENGTH 64
#define MAX_PATH_LENGTH 1024
#define MAX_TENSOR_NAME_LENGTH 64

nnfw_session::nnfw_session()
  : _subgraphs{nullptr}, _execution{nullptr},
    _kernel_registry{std::make_shared<onert::api::CustomKernelRegistry>()}, _tracing_ctx{nullptr}
{
  // DO NOTHING
}

nnfw_session::~nnfw_session() = default;

NNFW_STATUS nnfw_session::prepare()
{
  // NOTE. If users want to run prepare() more than one time, this could be removed.
  if (!isStateModelLoaded())
  {
    std::cerr << "Error during model prepare : ";
    if (isStateInitialized())
    {
      std::cerr << "prepare should be run once";
    }
    else
    {
      std::cerr << "invalid state";
    }
    std::cerr << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    _subgraphs.reset();
    std::shared_ptr<onert::exec::ExecutorMap> executors = _compiler->compile();
    _execution = std::make_unique<onert::exec::Execution>(executors);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model prepare : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _state = State::PREPARED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::run()
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::run : "
              << "run should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    _execution->execute();
  }
  catch (const onert::InsufficientBufferSizeException &e)
  {
    // Currently insufficient buffer always means output buffer.
    std::cerr << "Error during nnfw_session::run : " << e.what() << std::endl;
    return NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::run : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _state = State::FINISHED_RUN;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::run_async()
{
  if (!isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::run_async : "
              << "run_async should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  _execution->startExecute();

  _state = State::RUNNING;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::await()
{
  if (!isStateRunning())
  {
    std::cerr << "Error during nnfw_session::run_await : "
              << "run_await should be run after run_async" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _execution->waitFinish();

  _state = State::FINISHED_RUN;
  return NNFW_STATUS_NO_ERROR;
}

const onert::ir::Graph *nnfw_session::primary_subgraph()
{
  if (_subgraphs)
  {
    assert(!_execution);
    return _subgraphs->primary().get();
  }
  else
  {
    assert(_execution);
    // TODO Remove const_cast
    // We assumed the graph will not change after compilation, but shape could change
    return &_execution->primary_subgraph();
  }
}

bool nnfw_session::isStateInitialized()
{
  if (_state == State::INITIALIZED)
  {
    assert(!_subgraphs);
    assert(!_compiler);
    assert(!_execution);
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStateModelLoaded()
{
  if (_state == State::MODEL_LOADED)
  {
    assert(_subgraphs);
    assert(_compiler);
    assert(!_execution);
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStatePrepared()
{
  if (_state == State::PREPARED)
  {
    assert(!_subgraphs);
    assert(_compiler);
    assert(_execution);
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStateRunning()
{
  if (_state == State::RUNNING)
  {
    assert(!_subgraphs);
    assert(_compiler);
    assert(_execution);
    return true;
  }
  return false;
}

bool nnfw_session::isStateFinishedRun()
{
  if (_state == State::FINISHED_RUN)
  {
    assert(!_subgraphs);
    assert(_compiler);
    assert(_execution);
    return true;
  }
  else
  {
    return false;
  }
}

bool nnfw_session::isStatePreparedOrFinishedRun()
{
  return isStatePrepared() || isStateFinishedRun();
}
