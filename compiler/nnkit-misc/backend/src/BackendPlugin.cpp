/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nnkit/BackendPlugin.h"

#include <cassert>
#include <memory>
#include <iostream>

// NOTE dlfcn.h is not a standard library
#include <dlfcn.h>

namespace nnkit
{

BackendPlugin::BackendPlugin(BackendPlugin &&plugin)
{
  // Handle is transferd from 'binder' instance into this instance.
  _handle = plugin._handle;
  _entry = plugin._entry;

  plugin._handle = nullptr;
  plugin._entry = nullptr;
}

BackendPlugin::~BackendPlugin()
{
  if (_handle != nullptr)
  {
    dlclose(_handle);
  }
}

std::unique_ptr<Backend> BackendPlugin::create(const CmdlineArguments &args) const
{
  return _entry(args);
}

std::unique_ptr<BackendPlugin> make_backend_plugin(const std::string &path)
{
  if (path.empty())
  {
    throw std::runtime_error{"Backend library does not defined"};
  }

  void *handle;
  BackendPlugin::Entry entry;

  // NOTE Some backend (such as tflite) needs multithreading support (std::thread).
  //
  //      std::thread in libstdc++.so includes weak symbols for pthread_XXX functions,
  //      and these weak symbols should be overridden by strong symbols in libpthread.so.
  //      If not, std::thread will not work correctly.
  //
  //      RTLD_GLOBAL flag is necessary to allow weak symbols to be overridden.
  handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (handle == nullptr)
  {
    std::cerr << dlerror() << std::endl;
    exit(1);
  }

  char *error;
  entry = reinterpret_cast<BackendPlugin::Entry>(dlsym(handle, "make_backend"));
  if ((error = dlerror()) != nullptr)
  {
    dlclose(handle);
    std::cerr << error << std::endl;
    exit(1);
  }

  return std::make_unique<BackendPlugin>(handle, entry);
}

} // namespace nnkit
