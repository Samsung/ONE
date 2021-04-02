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

#include "Config.h"
#include "Helper.h"

#include <misc/string_helpers.h>

namespace onert
{
namespace api
{

namespace
{

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

} // namespace

Config::Config(nnfw_session *session) : _session{session}
{
  // DO NOTHING
}

NNFW_STATUS Config::setAvailableBackends(const char *backends)
{
  if (!_session->isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  try
  {
    if (!backends)
      return NNFW_STATUS_UNEXPECTED_NULL;
    if (null_terminating(backends, MAX_BACKEND_NAME_LENGTH) == false)
      return NNFW_STATUS_ERROR;

    auto &options = _session->_compiler->options();

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

NNFW_STATUS Config::setOpBackend(const char *op, const char *backend)
{
  if (!_session->isStateModelLoaded())
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

    auto &opcode_to_backend =
      _session->_compiler->options().manual_scheduler_options.opcode_to_backend;
    opcode_to_backend.emplace(onert::ir::toOpCode(key), backend);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during nnfw_session::set_op_backend : " << e.what() << std::endl;
    return NNFW_STATUS_ERROR;
  }
  return NNFW_STATUS_NO_ERROR;
}

NNFW_STATUS Config::setConfig(const char *key, const char *value)
{
  if (!_session->isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto &options = _session->_compiler->options();

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

NNFW_STATUS Config::getConfig(const char *key, char *value, size_t value_size)
{
  if (!_session->isStateModelLoaded())
    return NNFW_STATUS_INVALID_STATE;

  if (!key || !value)
    return NNFW_STATUS_UNEXPECTED_NULL;

  auto &options = _session->_compiler->options();

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

NNFW_STATUS Config::registerCustomOperation(const std::string &id, nnfw_custom_eval eval_func)
{
  _session->_kernel_registry->registerKernel(id, eval_func);
  return NNFW_STATUS_NO_ERROR;
}

} // namespace api
} // namespace onert
