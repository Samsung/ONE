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

#include <cstring>
#include <stdexcept>

// This should be called after _compiler is created
static bool only_for_cpu_backend(nnfw_session *session)
{
  char backends[128];
  NNFW_STATUS res = nnfw_query_info_str(session, NNFW_INFO_BACKENDS, backends);
  if (res == NNFW_STATUS_ERROR)
    throw std::runtime_error("error while calling nnfw_query_info_str() to get backends");

  // first backend should be "cpu"
  return (strcmp(backends, "cpu") == 0 || strstr(backends, "cpu;") == backends);
}

// This should be called after _compiler is created
static bool only_for_LinearExecutor(nnfw_session *session)
{
  char backends[128];
  NNFW_STATUS res = nnfw_query_info_str(session, NNFW_INFO_EXECUTOR, backends);
  if (res == NNFW_STATUS_ERROR)
    throw std::runtime_error("error while calling nnfw_query_info_str() to get executor");

  return (strcmp(backends, "Linear") == 0);
}

#endif // __NNFW_API_MODEL_TEST_HELPER_H__
