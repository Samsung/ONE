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

#ifndef __NNFW_API_HELPER_H__
#define __NNFW_API_HELPER_H__

#include "nnfw.h"

/*
 * API does not accept string argument longer than max length below
 */
#define MAX_BACKEND_NAME_LENGTH 32
#define MAX_OP_NAME_LENGTH 64
#define MAX_PATH_LENGTH 1024
#define MAX_TENSOR_NAME_LENGTH 64

#define NNFW_RETURN_ERROR_IF_NULL(p)      \
  do                                      \
  {                                       \
    if ((p) == NULL)                      \
      return NNFW_STATUS_UNEXPECTED_NULL; \
  } while (0)

namespace onert
{
namespace api
{

bool null_terminating(const char *str, uint32_t length);

}
} // namespace onert

#endif // __NNFW_API_HELPER_H__
