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
#include "backend/controlflow/Config.h"
#include "backend/IConfig.h"
#include "util/logging.h"
#include "util/ConfigSource.h"
#include "misc/string_helpers.h"

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

BackendManager::BackendManager() { loadControlflowBackend(); }

void BackendManager::loadControlflowBackend()
{
  auto backend_object = std::unique_ptr<backend::controlflow::Backend, backend_destroy_t>(
      new backend::controlflow::Backend, [](backend::Backend *backend) { delete backend; });

  bool initialized = backend_object->config()->initialize(); // Call initialize here?
  if (!initialized)
  {
    throw std::runtime_error(backend::controlflow::Config::ID + " backend initialization failed");
  }
  _controlflow = backend_object.get(); // Save the controlflow backend implementation pointer
  assert(_controlflow);
  _gen_map.emplace(backend_object->config()->id(), std::move(backend_object));
}

void BackendManager::loadBackend(const std::string &backend)
{
  if (get(backend) != nullptr)
  {
    return;
  }

  // TODO Remove indentation
  {
    const std::string backend_boost_so = "libbackend_" + backend + "-boost" + SHARED_LIB_EXT;
    const std::string backend_so = "libbackend_" + backend + SHARED_LIB_EXT;

    void *handle = dlopen(backend_boost_so.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr)
    {
      handle = dlopen(backend_so.c_str(), RTLD_LAZY | RTLD_LOCAL);

      if (handle == nullptr)
      {
        VERBOSE_F() << "Failed to load backend '" << backend << "' - " << dlerror() << std::endl;
        return;
      }

      VERBOSE_F() << "Successfully loaded '" << backend << "' - " << backend_so << "\n";
    }
    else
    {
      VERBOSE_F() << "Successfully loaded '" << backend << "' - " << backend_boost_so << "\n";
    }

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
      bool initialized = backend_object->config()->initialize(); // Call initialize here?
      if (!initialized)
      {
        VERBOSE_F() << backend.c_str() << " backend initialization failed. Don't use this backend"
                    << std::endl;
        dlclose(handle);
        return;
      }
      _gen_map.emplace(backend_object->config()->id(), std::move(backend_object));
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

const backend::controlflow::Backend *BackendManager::getControlflow() const { return _controlflow; }

} // namespace compiler
} // namespace onert
