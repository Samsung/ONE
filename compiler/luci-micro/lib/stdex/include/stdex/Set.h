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

#ifndef __STDEX_SET_H__
#define __STDEX_SET_H__

#include <set>

template <typename T> bool operator==(const std::set<T> &lhs, const std::set<T> &rhs)
{
  if (rhs.size() != lhs.size())
  {
    return false;
  }

  for (const auto &element : lhs)
  {
    if (rhs.find(element) == rhs.end())
    {
      return false;
    }
  }

  return true;
}

template <typename T> std::set<T> operator-(const std::set<T> &lhs, const std::set<T> &rhs)
{
  std::set<T> res;

  for (const auto &element : lhs)
  {
    if (rhs.find(element) == rhs.end())
    {
      res.insert(element);
    }
  }

  return res;
}

#endif // __STDEX_SET_H__
