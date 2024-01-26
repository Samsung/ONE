/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_TRAIN_NNFW_UTIL_H__
#define __ONERT_TRAIN_NNFW_UTIL_H__

#include "nnfw.h"
#include "nnfw_experimental.h"

#include <ostream>

#define NNPR_ENSURE_STATUS(a)        \
  do                                 \
  {                                  \
    if ((a) != NNFW_STATUS_NO_ERROR) \
    {                                \
      exit(-1);                      \
    }                                \
  } while (0)

namespace onert_train
{
uint64_t num_elems(const nnfw_tensorinfo *ti);
uint64_t bufsize_for(const nnfw_tensorinfo *ti);

std::string toString(NNFW_TRAIN_OPTIMIZER opt);
std::string toString(NNFW_TRAIN_LOSS loss);
std::string toString(NNFW_TRAIN_LOSS_REDUCTION loss_rdt);
std::ostream &operator<<(std::ostream &os, const nnfw_train_info &info);
} // end of namespace onert_train
#endif // __ONERT_TRAIN_NNFW_UTIL_H__
