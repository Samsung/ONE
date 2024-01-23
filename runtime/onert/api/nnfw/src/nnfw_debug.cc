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

#include <util/ConfigSource.h>

#define NNFW_RETURN_ERROR_IF_NULL(p)      \
  do                                      \
  {                                       \
    if ((p) == NULL)                      \
      return NNFW_STATUS_UNEXPECTED_NULL; \
  } while (0)

NNFW_STATUS nnfw_set_config(nnfw_session *session, const char *key, const char *value)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->set_config(key, value);
}

NNFW_STATUS nnfw_get_config(nnfw_session *session, const char *key, char *value, size_t value_size)
{
  NNFW_RETURN_ERROR_IF_NULL(session);
  return session->get_config(key, value, value_size);
}
