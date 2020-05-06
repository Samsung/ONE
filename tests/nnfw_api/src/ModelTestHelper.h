/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_API_MODEL_TEST_HELPER_H__
#define __NNFW_API_MODEL_TEST_HELPER_H__

#include <nnfw_debug.h>
#include <util/ConfigSource.h>
#include <misc/string_helpers.h>

#include <cstring>
#include <stdexcept>

// This should be called after _compiler is created
static bool onlyForCpuBackend(nnfw_session *session)
{
  char backends[128];
  NNFW_STATUS res =
      nnfw_get_config(session, onert::util::config::BACKENDS, backends, sizeof(backends));
  if (res == NNFW_STATUS_ERROR)
    throw std::runtime_error("error while calling nnfw_query_info_str() to get backends");

  auto backend_list = nnfw::misc::split(std::string(backends), ';');
  // first backend should be "cpu"
  return (backend_list.size() > 0 && backend_list.at(0) == "cpu");
}

// This should be called after _compiler is created
static bool onlyForLinearExecutor(nnfw_session *session)
{
  char executor[128];
  NNFW_STATUS res =
      nnfw_get_config(session, onert::util::config::EXECUTOR, executor, sizeof(executor));
  if (res == NNFW_STATUS_ERROR)
    throw std::runtime_error("error while calling nnfw_query_info_str() to get executor");

  return (strcmp(executor, "Linear") == 0);
}

#endif // __NNFW_API_MODEL_TEST_HELPER_H__
