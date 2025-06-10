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

#ifndef __KBENCHMARK_OPERATION_LOADER_H__
#define __KBENCHMARK_OPERATION_LOADER_H__

#include <string>
#include <unordered_map>

#include "Operation.h"
#include "operations/Convolution.h"
#include "operations/TransposeConv.h"

namespace kbenchmark
{

class OperationLoader
{
public:
  static OperationLoader &getInstance(void)
  {
    static OperationLoader instance;
    return instance;
  }

  Operation *operator[](const std::string &name) { return _map[name]; }
  bool is_valid(const std::string &name) { return _map.count(name); }

private:
  OperationLoader(void)
  {
#define OP(ConfigName, OperationName) _map[ConfigName] = new operation::OperationName();
#include "Operations.lst"
#undef OP
  }

  ~OperationLoader() = default;
  OperationLoader(const OperationLoader &) = delete;
  OperationLoader &operator=(const OperationLoader &) = delete;

private:
  std::unordered_map<std::string, Operation *> _map;
};

} // namespace kbenchmark

#endif // __KBENCHMARK_OPERATION_LOADER_H__
