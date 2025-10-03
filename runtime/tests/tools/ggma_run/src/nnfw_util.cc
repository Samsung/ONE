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

#include "nnfw.h"
#include "nnfw_util.h"

#include <cassert>
#include <iostream>
#include <string>

#define NNPR_ENSURE_STATUS(a)        \
  do                                 \
  {                                  \
    if ((a) != NNFW_STATUS_NO_ERROR) \
    {                                \
      exit(-1);                      \
    }                                \
  } while (0)

namespace ggma_run
{
void print_version()
{
  uint32_t version;
  NNPR_ENSURE_STATUS(nnfw_query_info_u32(NULL, NNFW_INFO_ID_VERSION, &version));
  std::cout << "ggma_run (nnfw runtime: v" << (version >> 24) << "."
            << ((version & 0x0000FF00) >> 8) << "." << (version & 0xFF) << ")" << std::endl;
}

} // namespace ggma_run
