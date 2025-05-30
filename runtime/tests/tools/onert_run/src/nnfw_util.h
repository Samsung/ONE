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

#ifndef __ONERT_RUN_NNFW_UTIL_H__
#define __ONERT_RUN_NNFW_UTIL_H__

#include "nnfw.h"

#define NNPR_ENSURE_STATUS(a)        \
  do                                 \
  {                                  \
    if ((a) != NNFW_STATUS_NO_ERROR) \
    {                                \
      exit(-1);                      \
    }                                \
  } while (0)

namespace onert_run
{
uint64_t num_elems(const nnfw_tensorinfo *ti);
uint64_t bufsize_for(const nnfw_tensorinfo *ti);
uint64_t has_dynamic_dim(const nnfw_tensorinfo *ti);
} // end of namespace onert_run

#endif // __ONERT_RUN_NNFW_UTIL_H__
