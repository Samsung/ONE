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

#ifndef __ONERT_COMPILER_BACKEND_MANAGER_H__
#define __ONERT_COMPILER_BACKEND_MANAGER_H__

#include "backend/Backend.h"
#include "ir/Operands.h"

#include <map>
#include <memory>

namespace onert::compiler
{

class BackendManager
{
public:
  using backend_create_t = backend::Backend *(*)();
  using backend_destroy_t = void (*)(backend::Backend *);
  using dlhandle_destroy_t = std::function<void(void *)>;

  static BackendManager &get();

public:
  backend::Backend *get(std::string_view key);
  const backend::Backend *get(std::string_view key) const;
  const backend::Backend *getBuiltin() const;
  const std::vector<const backend::Backend *> getAll() const
  {
    std::vector<const backend::Backend *> v;
    for (const auto &p : _gen_map)
      v.emplace_back(p.second.get());
    return v;
  }
  size_t num_backends() const { return _gen_map.size(); }
  /**
   * @brief load backend plugin
   *
   * @param backend backend to be loaded
   *
   * @return
   */
  void loadBackend(const std::string &backend);

private:
  BackendManager();

private:
  std::map<std::string, std::unique_ptr<void, dlhandle_destroy_t>> _handle_map;
  std::map<std::string, std::unique_ptr<backend::Backend, backend_destroy_t>, std::less<>> _gen_map;
  backend::Backend *_builtin{nullptr};
  /**
   * @brief load builtin backend
   *
   * @param backend backend to be loaded
   *
   * @return
   */
  void loadBuiltinBackend();
};

} // namespace onert::compiler

#endif // __ONERT_COMPILER_BACKEND_MANAGER_H__
