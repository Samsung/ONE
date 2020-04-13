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

#ifndef __KBENCHMARK_UTILS_H__
#define __KBENCHMARK_UTILS_H__

#include <sstream>
#include <string>
#include <vector>
#include <cassert>

namespace kbenchmark
{

void check_valid_key(const std::string &key, OperationInfo &info)
{
  OperationInfo::const_iterator it;
  it = info.find(key);
  assert(it != info.end());
}

std::vector<int> dims(const std::string &src)
{
  std::vector<int> dim;

  std::stringstream ss(src);
  int i;
  while (ss >> i)
  {
    dim.push_back(i);
    if (ss.peek() == ',')
      ss.ignore();
  }
  return dim;
}

std::vector<int> get_key_dims(const std::string &key, OperationInfo &info)
{
  check_valid_key(key, info);
  return dims(info[key]);
}

int get_key_int(const std::string &key, OperationInfo &info)
{
  check_valid_key(key, info);
  return std::stoi(info[key]);
}

std::string get_key_string(const std::string &key, OperationInfo &info)
{
  check_valid_key(key, info);
  return info[key];
}

} // namespace kbenchmark

#endif // __KBENCHMARK_UTILS_H__
