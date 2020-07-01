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

#include "souschef/LexicalCast.h"

#include <cassert>
#include <limits>

namespace souschef
{

template <> float to_number(const std::string &s) { return std::stof(s); }
template <> int to_number(const std::string &s) { return std::stoi(s); }
template <> int64_t to_number(const std::string &s) { return std::stoll(s); }
template <> uint8_t to_number(const std::string &s)
{
  int temp = std::stoi(s);
  assert(temp >= 0);
  assert(temp <= std::numeric_limits<uint8_t>::max());
  return static_cast<uint8_t>(temp);
}
template <> bool to_number(const std::string &s)
{
  if (std::stoi(s) || s == "T" || s == "t" || s == "TRUE" || s == "true")
    return true;
  return false;
}

} // namespace souschef
