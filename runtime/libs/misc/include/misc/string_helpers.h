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

/**
 * @file string_helpers.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains helper functions for std::string
 */

#include <ostream>
#include <string>
#include <sstream>
#include <vector>

namespace
{

template <typename Arg> void _str(std::ostream &os, Arg &&arg) { os << std::forward<Arg>(arg); }

template <typename Arg, typename... Args> void _str(std::ostream &os, Arg &&arg, Args &&...args)
{
  _str(os, std::forward<Arg>(arg));
  _str(os, std::forward<Args>(args)...);
}

} // namespace

namespace nnfw
{
namespace misc
{

inline std::vector<std::string> split(const std::string &s, char delim)
{
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> elems;
  while (std::getline(ss, item, delim))
  {
    elems.push_back(item);
  }
  return elems;
}

template <typename... Args> std::string str(Args &&...args)
{
  std::stringstream ss;
  _str(ss, std::forward<Args>(args)...);
  return ss.str();
}

template <typename InputIt> std::string join(InputIt first, InputIt last, const std::string &concat)
{
  std::string ret;
  if (first == last)
    return ret;

  ret += *first;
  for (++first; first != last; ++first)
  {
    ret += concat;
    ret += *first;
  }
  return ret;
}

} // namespace misc
} // namespace nnfw
