/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ENCO_STRING_H__
#define __ENCO_STRING_H__

//
// String-manipulating routines
//
#include <ostream>
#include <sstream>

#include <string>

namespace enco
{

template <typename It> void concat(std::ostream &os, const std::string &sep, It beg, It end)
{
  uint32_t count = 0;

  for (auto it = beg; it != end; ++it, ++count)
  {
    if (count == 0)
    {
      os << *it;
    }
    else
    {
      os << sep << *it;
    }
  }
}

template <typename It> std::string concat(const std::string &sep, It beg, It end)
{
  std::stringstream ss;
  concat(ss, sep, beg, end);
  return ss.str();
}

} // namespace enco

#endif // __ENCO_STRING_H__
