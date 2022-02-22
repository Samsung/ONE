/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "pepper/csv2vec.h"

#include <algorithm>
#include <sstream>
#include <cassert>

namespace pepper
{

template <> std::vector<std::string> csv_to_vector(const std::string &str)
{
  std::vector<std::string> ret;
  std::istringstream is(str);
  for (std::string item; std::getline(is, item, ',');)
  {
    ret.push_back(item);
  }
  return ret;
}

// TODO merge std::string and int32_t type

template <> std::vector<int32_t> csv_to_vector(const std::string &str)
{
  std::vector<int32_t> ret;
  std::istringstream is(str);
  for (int32_t i; is >> i;)
  {
    assert(i != ',');
    ret.push_back(i);
    if (is.peek() == ',')
      is.ignore();
  }
  return ret;
}

template <> bool is_one_of(const std::string &item, const std::vector<std::string> &items)
{
  return std::find(items.begin(), items.end(), item) != items.end();
}

} // namespace pepper
