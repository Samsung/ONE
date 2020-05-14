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

#include "compiler/BackendManager.h"

#include <memory>
#include <dlfcn.h>

#include "backend/Backend.h"
#include "backend/controlflow/Backend.h"
#include "backend/IConfig.h"
#include "util/logging.h"
#include "util/ConfigSource.h"
#include "misc/string_helpers.h"

namespace onert
{
namespace compiler
{

BackendManager &BackendManager::get()
{
  static BackendManager object;
  return object;
}

template <typename T, class... Types>
void BackendManager::loadObjectFromPlugin(std::shared_ptr<T> &object_of_plugin_class,
                                          const std::string obj_creator_func_name, void *handle,
                                          Types &&... args)
{
  T *(*allocate_obj)(Types && ... Args);
  // load object creator function
  allocate_obj = (T * (*)(Types && ... Args))dlsym(handle, obj_creator_func_name.c_str());
  if (allocate_obj == nullptr)
  {
    fprintf(stderr, "BackendManager: unable to open function %s: %s\n",
            obj_creator_func_name.c_str(), dlerror());
    abort();
  }

  object_of_plugin_class.reset(allocate_obj(args...));
}

void BackendManager::loadBackend(const std::string &backend)
{
  if (get(backend) != nullptr)
  {
    return;
  }

  if (backend == backend::controlflow::Config::ID)
  {
    // TODO Make this backend to be loaded from .so file
    // Add controlflow Backend
    auto backend_create = []() -> backend::Backend * {
      return new onert::backend::controlflow::Backend;
    };

    auto backend_destroy = [](backend::Backend *backend) { delete backend; };

    auto backend_object =
        std::unique_ptr<backend::Backend, backend_destroy_t>(backend_create(), backend_destroy);
    auto backend_object_raw = backend_object.get();
    bool initialized = backend_object->config()->initialize(); // Call initialize here?
    if (!initialized)
    {
      throw std::runtime_error(backend::controlflow::Config::ID + " backend initialization failed");
    }
    (void)backend_object_raw;
    _gen_map.emplace(backend_object->config()->id(), std::move(backend_object));
    _available_backends.push_back(backend_object_raw);
  }
  else
  {
    const std::string backend_plugin = "libbackend_" + backend + ".so";
    void *handle = dlopen(backend_plugin.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr)
    {
      VERBOSE(BackendManager::loadBackend) << "loadBackend failed to load plugin of "
                                           << backend.c_str() << " backend: " << dlerror()
                                           << std::endl;
      return;
    }

    VERBOSE(BackendManager::loadBackend) << "loaded " << backend_plugin << " as a plugin of "
                                         << backend << " backend\n";

    {
      // load object creator function
      auto backend_create = (backend_create_t)dlsym(handle, "onert_backend_create");
      if (backend_create == nullptr)
      {
        fprintf(stderr, "BackendManager: unable to open function onert_backend_create : %s\n",
                dlerror());
        abort();
      }

      // load object creator function
      auto backend_destroy = (backend_destroy_t)dlsym(handle, "onert_backend_destroy");
      if (backend_destroy == nullptr)
      {
        fprintf(stderr, "BackendManager: unable to open function onert_backend_destroy : %s\n",
                dlerror());
        abort();
      }

      auto backend_object =
          std::unique_ptr<backend::Backend, backend_destroy_t>(backend_create(), backend_destroy);
      auto backend_object_raw = backend_object.get();
      bool initialized = backend_object->config()->initialize(); // Call initialize here?
      if (!initialized)
      {
        VERBOSE(BackendManager::loadBackend)
            << backend.c_str() << " backend initialization failed. Don't use this backend"
            << std::endl;
        dlclose(handle);
        return;
      }
      _gen_map.emplace(backend_object->config()->id(), std::move(backend_object));
      _available_backends.push_back(backend_object_raw);
    }

    // Save backend handle (avoid warning by handle lost without dlclose())
    auto u_handle = std::unique_ptr<void, dlhandle_destroy_t>{handle, [](void *h) { dlclose(h); }};
    _handle_map.emplace(backend, std::move(u_handle));
  }
}

backend::Backend *BackendManager::get(const std::string &key)
{
  if (_gen_map.find(key) != _gen_map.end())
  {
    return _gen_map.at(key).get();
  }

  return nullptr;
}

const backend::Backend *BackendManager::get(const std::string &key) const
{
  if (_gen_map.find(key) != _gen_map.end())
  {
    return _gen_map.at(key).get();
  }

  return nullptr;
}

const backend::Backend *BackendManager::getDefault() const { return get("cpu"); }

} // namespace compiler
} // namespace onert
