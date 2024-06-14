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

#include "../backend/builtin/Backend.h"
#include "../backend/builtin/Config.h"

#include <dlfcn.h>
#include <memory>

static const char *SHARED_LIB_EXT =
#if defined(__APPLE__) && defined(__MACH__)
  ".dylib";
#else
  ".so";
#endif

namespace onert
{
namespace compiler
{

BackendManager &BackendManager::get()
{
  static BackendManager object;
  return object;
}

BackendManager::BackendManager() { loadBuiltinBackend(); }

void BackendManager::loadBuiltinBackend()
{
  auto backend_object = std::unique_ptr<backend::builtin::Backend, backend_destroy_t>(
    new backend::builtin::Backend, [](backend::Backend *backend) { delete backend; });

  bool initialized = backend_object->config()->initialize(); // Call initialize here?
  if (!initialized)
  {
    throw std::runtime_error(backend::builtin::Config::ID + " backend initialization failed");
  }
  _builtin = backend_object.get(); // Save the builtin backend implementation pointer
  assert(_builtin);
  _gen_map.emplace(backend_object->config()->id(), std::move(backend_object));
}

void BackendManager::loadBackend(const std::string &backend)
{
  if (get(backend) != nullptr)
  {
    return;
  }

  const std::string backend_so = "libbackend_" + backend + SHARED_LIB_EXT;
#ifdef __ANDROID__
  void *handle = dlopen(backend_so.c_str(), RTLD_LAZY | RTLD_LOCAL);
#else
  void *handle = dlopen(backend_so.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
#endif

  if (handle == nullptr)
  {
    VERBOSE(BackendManager) << "Failed to load backend '" << backend << "' - " << dlerror() << "\n";
    return;
  }

  VERBOSE(BackendManager) << "Successfully loaded '" << backend << "'(" << backend_so << ")\n";

  {
    // load object creator function
    auto backend_create = (backend_create_t)dlsym(handle, "onert_backend_create");
    if (backend_create == nullptr)
    {
      // TODO replace `fprintf` with `VERBOSE`
      fprintf(stderr, "BackendManager: unable to find function `onert_backend_create` : %s\n",
              dlerror());
      dlclose(handle);
      return;
    }

    // load object creator function
    auto backend_destroy = (backend_destroy_t)dlsym(handle, "onert_backend_destroy");
    if (backend_destroy == nullptr)
    {
      // TODO replace `fprintf` with `VERBOSE`
      fprintf(stderr, "BackendManager: unable to find `function onert_backend_destroy` : %s\n",
              dlerror());
      dlclose(handle);
      return;
    }

    auto backend_object =
      std::unique_ptr<backend::Backend, backend_destroy_t>(backend_create(), backend_destroy);
    bool initialized = backend_object->config()->initialize(); // Call initialize here?
    if (!initialized)
    {
      VERBOSE(BackendManager) << backend.c_str()
                              << " backend initialization failed. Don't use this backend"
                              << std::endl;
      dlclose(handle);
      return;
    }
    _gen_map.emplace(backend_object->config()->id(), std::move(backend_object));
  }

  // Save backend handle (avoid warning by handle lost without dlclose())
  auto u_handle = std::unique_ptr<void, dlhandle_destroy_t>{
    handle, [id = backend, filename = backend_so](void *h) {
      if (dlclose(h) == 0)
      {
        VERBOSE(BackendManager) << "Successfully unloaded '" << id << "'(" << filename << ")\n";
      }
      else
      {
        VERBOSE(BackendManager) << "Failed to unload backend '" << id << "'- " << dlerror() << "\n";
      }
    }};
  _handle_map.emplace(backend, std::move(u_handle));
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

const backend::Backend *BackendManager::getBuiltin() const { return _builtin; }

} // namespace compiler
} // namespace onert
