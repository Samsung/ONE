/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RandomString.h"

#include <cstdlib>
#include <string>

namespace fme_apply
{

std::string random_str(uint32_t len)
{
  static const char cand[] = "0123456789"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                             "abcdefghijklmnopqrstuvwxyz";

  std::string res;
  res.reserve(len);

  for (uint32_t i = 0; i < len; ++i)
  {
    res += cand[std::rand() % (sizeof(cand) - 1)];
  }

  return res;
}

} // namespace fme_apply
