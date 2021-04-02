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

namespace
{

// Is null-terminating in length ?
bool null_terminating(const char *str, uint32_t length)
{
  for (uint32_t i = 0; i < length; i++)
  {
    if (*(str + i) == '\0')
    {
      return true;
    }
  }
  return false;
}

} // namespace

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

NNFW_STATUS nnfw_session::register_custom_operation(const std::string &id,
                                                    nnfw_custom_eval eval_func)
{
  _kernel_registry->registerKernel(id, eval_func);
  return NNFW_STATUS_NO_ERROR;
}

static std::string get_op_backend_string(std::string op)
{
#define MAP_MACRO(CircleName, OneRTName) {#CircleName, #OneRTName},

  static std::unordered_map<std::string, std::string> operation_map = {
#include "OpMap.lst"
  };

#undef MAP_MACRO

  auto n = operation_map.find(op);

  if (n == operation_map.end())
  {
    // this return value is handled by a caller to return error code
    return std::string("");
  }
  else
  {
    return n->second;
  }
}

NNFW_STATUS nnfw_session::set_available_backends(const char *backends)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (!backends)
      return NNFW_STATUS_UNEXPECTED_NULL;
    if (null_terminating(backends, MAX_BACKEND_NAME_LENGTH) == false)
      return NNFW_STATUS_ERROR;

    auto &options = _compiler->options();

    using namespace onert::util;

    options.backend_list = nnfw::misc::split(std::string{backends}, ';');
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_available_backends : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_op_backend(const char *op, const char *backend)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (!op || !backend)
      return NNFW_STATUS_UNEXPECTED_NULL;
    if (!null_terminating(op, MAX_OP_NAME_LENGTH) ||
        !null_terminating(backend, MAX_BACKEND_NAME_LENGTH))
      return NNFW_STATUS_ERROR;

    auto key = get_op_backend_string(op);

    if (key.empty())
    {
      return NNFW_STATUS_ERROR;
    }

    auto &opcode_to_backend = _compiler->options().manual_scheduler_options.opcode_to_backend;
    opcode_to_backend.emplace(onert::ir::toOpCode(key), backend);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_op_backend : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS nnfw_session::set_config(const char *key, const char *value)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto &options = _compiler->options();

  using namespace onert::util;

  const std::string skey = key;

  if (skey == config::TRACE_FILEPATH)
  {
    options.trace_filepath = value;
  }
  else if (skey == config::GRAPH_DOT_DUMP)
  {
    options.graph_dump_level = toInt(value);
  }
  else if (skey == config::EXECUTOR)
  {
    options.executor = value;
  }
  else if (skey == config::OP_BACKEND_ALLOPS)
  {
    options.manual_scheduler_options.backend_for_all = value;
  }
  else if (skey == config::USE_SCHEDULER)
  {
    options.he_scheduler = toBool(value);
  }
  else if (skey == config::PROFILING_MODE)
  {
    options.he_profiling_mode = toBool(value);
  }
  else if (skey == config::DISABLE_COMPILE)
  {
    options.disable_compile = toBool(value);
  }
  else
  {
    return NNFW_STATUS_ERROR;
  }
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

NNFW_STATUS nnfw_session::get_config(const char *key, char *value, size_t value_size)
{
  if (!isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto &options = _compiler->options();

  auto check_boundary = [](size_t dest_size, std::string &src) {
    if (dest_size < src.length() + 1 /* for '\0' */)
    {
      std::cerr << "buffer is small to copy config value." << std::endl;
      return false;
    }
    return true;
  };

  const std::string skey = key;

  if (skey == onert::util::config::BACKENDS)
  {
    if (options.backend_list.size() == 0)
      return NNFW_STATUS_NO_ERROR; // no setting backend is not an error of get_config_str()

    auto str = nnfw::misc::join(options.backend_list.begin(), options.backend_list.end(), ";");

    if (!check_boundary(value_size, str))
      return NNFW_STATUS_ERROR;

    strncpy(value, str.c_str(), value_size);
  }
  else if (skey == onert::util::config::EXECUTOR)
  {
    if (!check_boundary(value_size, options.executor))
      return NNFW_STATUS_ERROR;

    strncpy(value, options.executor.c_str(), options.executor.length());
  }
  else
  {
    return NNFW_STATUS_ERROR;
  }

  return NNFW_STATUS_NO_ERROR;
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
