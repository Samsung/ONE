/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_COMPILER_BACKEND_RESOLVER_H__
#define __ONERT_COMPILER_BACKEND_RESOLVER_H__

#include <unordered_map>
#include <typeindex>

#include "backend/Backend.h"
#include "ir/OperationIndexMap.h"

namespace onert::compiler
{

class BackendResolver
{
public:
  const backend::Backend *getBackend(const ir::OperationIndex &index) const
  {
    return _gen_map.at(index);
  }

  void setBackend(const ir::OperationIndex &index, const backend::Backend *backend)
  {
    _gen_map[index] = backend;
  }

  bool hasBackend(const ir::OperationIndex &index) const
  {
    return _gen_map.find(index) != _gen_map.end();
  }

  void
  iterate(const std::function<void(const ir::OperationIndex &, const backend::Backend &)> &fn) const
  {
    for (const auto &[op_index, backend] : _gen_map)
    {
      fn(op_index, *backend);
    }
  }

private:
  ir::OperationIndexMap<const backend::Backend *> _gen_map;
};

} // namespace onert::compiler

#endif // __ONERT_COMPILER_BACKEND_RESOLVER_H__
