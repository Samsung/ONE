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

#include "nnfw_session.h"

#include <util/logging.h>

#include <misc/string_helpers.h>

#include <iostream>
#include <string>
#include <vector>

nnfw_session::nnfw_session()
  : _subgraphs{nullptr}, _execution{nullptr},
    _kernel_registry{std::make_shared<onert::api::CustomKernelRegistry>()}, _tracing_ctx{nullptr}
{
  // DO NOTHING
}

nnfw_session::~nnfw_session() = default;

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
