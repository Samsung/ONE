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

#ifndef __SOUSCHEF_REGISTRY_H__
#define __SOUSCHEF_REGISTRY_H__

#include <map>
#include <memory>
#include <string>

namespace souschef
{

template <typename T> class Registry
{
public:
  void add(const std::string &name, std::unique_ptr<T> &&entry)
  {
    _content[name] = std::move(entry);
  }

  const T &lookup(const std::string &name) const { return *(_content.at(name)); }

private:
  std::map<std::string, std::unique_ptr<T>> _content;
};

} // namespace souschef

#endif // __SOUSCHEF_REGISTRY_H__
