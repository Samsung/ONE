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
#include <stdexcept>

namespace souschef
{

template <> float to_number(const std::string &s) { return std::stof(s); }
template <> int to_number(const std::string &s) { return std::stoi(s); }
template <> int16_t to_number(const std::string &s)
{
  // There are no standard function to parse int16_t or short int
  // This function simulates behavior similar stoi, stol and stoll
  int res = std::stol(s);
  // standard does not specify string in error message, this is arbitrary
  if (res < std::numeric_limits<int16_t>::min() || res > std::numeric_limits<int16_t>::max())
  {
    throw std::out_of_range("to_number<int16_t>");
  }
  return res;
}
template <> int64_t to_number(const std::string &s) { return std::stoll(s); }
template <> uint8_t to_number(const std::string &s)
{
  int temp = std::stoi(s);
  assert(temp >= 0);
  assert(temp <= std::numeric_limits<uint8_t>::max());
  return static_cast<uint8_t>(temp);
}
template <> int8_t to_number(const std::string &s)
{
  int temp = std::stoi(s);
  assert(temp >= std::numeric_limits<int8_t>::min());
  assert(temp <= std::numeric_limits<int8_t>::max());
  return static_cast<int8_t>(temp);
}
template <> bool to_number(const std::string &s)
{
  if (s == "T" || s == "t" || s == "TRUE" || s == "true" || s == "1")
    return true;
  if (s == "F" || s == "f" || s == "FALSE" || s == "false" || s == "0")
    return false;
  throw std::invalid_argument("Unsupported boolean argument");
}
template <> std::string to_number(const std::string &s) { return s; }

} // namespace souschef
