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

#include <memory>
#include <map>

#include "ir/Operands.h"
#include "backend/Backend.h"

namespace onert
{
namespace compiler
{

class BackendManager
{
public:
  using backend_create_t = backend::Backend *(*)();
  using backend_destroy_t = void (*)(backend::Backend *);
  using dlhandle_destroy_t = void (*)(void *);

  static BackendManager &get();

public:
  backend::Backend *get(const std::string &key);
  const backend::Backend *get(const std::string &key) const;
  const backend::Backend *getControlflow() const;
  const std::vector<const backend::Backend *> &getAll() const { return _available_backends; };
  /**
   * @brief load backend plugin
   *
   * @param backend backend to be loaded
   *
   * @return
   */
  void loadBackend(const std::string &backend);

private:
  BackendManager() = default;

private:
  std::vector<const backend::Backend *> _available_backends;
  std::map<std::string, std::unique_ptr<void, dlhandle_destroy_t>> _handle_map;
  std::map<std::string, std::unique_ptr<backend::Backend, backend_destroy_t>> _gen_map;
  /**
   * @brief Allocate an object of a class of a plugin by loading a plugin function, that does
   * allocation, and calling it
   *
   * @param object_of_plugin_class target object
   * @param obj_creator_func_name name of the plugin function, that allocates an object
   * @param handle handle of the plugin
   * @param args arguments to pass to constructor of the plugin class
   *
   * @return
   */
  template <typename T, class... Types>
  void loadObjectFromPlugin(std::shared_ptr<T> &object_of_plugin_class,
                            const std::string obj_creator_func_name, void *handle,
                            Types &&... args);

  /**
   * @brief load controlflow backend
   *
   * @param backend backend to be loaded
   *
   * @return
   */
  void loadControlflowBackend();
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_BACKEND_MANAGER_H__
