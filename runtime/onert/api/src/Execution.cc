/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Execution.h"

#include <util/Exceptions.h>

namespace onert
{
namespace api
{

Execution::Execution(nnfw_session *session) : _session{session}
{
  // DO NOTHING
}

NNFW_STATUS Execution::prepare()
{
  // NOTE. If users want to run prepare() more than one time, this could be removed.
  if (!_session->isStateModelLoaded())
  {
    std::cerr << "Error during model prepare : ";
    if (_session->isStateInitialized())
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
    _session->_subgraphs.reset();
    std::shared_ptr<onert::exec::ExecutorMap> executors = _session->_compiler->compile();
    _session->_execution = std::make_unique<onert::exec::Execution>(executors);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during model prepare : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _session->_state = nnfw_session::State::PREPARED;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS Execution::run()
{
  if (!_session->isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::run : "
              << "run should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  try
  {
    _session->_execution->execute();
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

  _session->_state = nnfw_session::State::FINISHED_RUN;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS Execution::runAsync()
{
  if (!_session->isStatePreparedOrFinishedRun())
  {
    std::cerr << "Error during nnfw_session::run_async : "
              << "run_async should be run after prepare" << std::endl;
    return NNFW_STATUS_INVALID_STATE;
  }

  _session->_execution->startExecute();

  _session->_state = nnfw_session::State::RUNNING;
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS Execution::await()
{
  if (!_session->isStateRunning())
  {
    std::cerr << "Error during nnfw_session::run_await : "
              << "run_await should be run after run_async" << std::endl;
    return NNFW_STATUS_ERROR;
  }

  _session->_execution->waitFinish();

  _session->_state = nnfw_session::State::FINISHED_RUN;
  return NNFW_STATUS_NO_ERROR;
}

} // namespace api
} // namespace onert
